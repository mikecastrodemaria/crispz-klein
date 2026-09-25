"""LoRA posee sous offload (crispz-klein 1.36.6).

Deux LoRA DoRA choisies comme LoRA d'edition mettaient fin a la session : le
chargement echouait sur "Cannot copy out of meta tensor", puis TOUS les rendus
suivants sur "Cannot generate a cpu tensor from a generator of type cuda".
Couvre :
  - les LoRA sont chargees avec low_cpu_mem_usage=False (pas de couche sur 'meta',
    une cle filtree garde sa valeur d'init) ;
  - un chargement qui echoue apres que diffusers a retire les hooks d'offload les
    fait remettre, au lieu de laisser le pipe sur le CPU ;
  - restore_offload repare un pipe laisse sur le CPU et ne touche a rien sinon.

Ni GPU ni modele : le pipe est simule, les LoRA sont des fichiers minuscules.

Run:  .venv/Scripts/python tests/test_lora_offload.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_pipeline as cz  # noqa: E402

META = ("Cannot copy out of meta tensor; no data! Please use "
        "torch.nn.Module.to_empty() instead of torch.nn.Module.to()")


class FakePipe:
    """Pipe minimal au comportement de diffusers : charger une LoRA retire d'abord
    les hooks d'offload, et ne les remet qu'en cas de succes."""

    def __init__(self, hooks=True, fail=False):
        self._all_hooks = ["hook"] if hooks else []
        self._execution_device = torch.device("cuda" if hooks else "cpu")
        self.fail = fail
        self.loaded, self.enabled, self.adapters, self.cleared = [], 0, None, 0

    def load_lora_weights(self, *a, **kw):
        self._all_hooks = []                       # diffusers retire l'offload
        self._execution_device = torch.device("cpu")
        self.loaded.append((a, kw))
        if self.fail:
            raise RuntimeError(META)
        self.enable_model_cpu_offload()             # ... et le remet si tout va bien

    def enable_model_cpu_offload(self, *a, **kw):
        self._all_hooks = ["hook"]
        self._execution_device = torch.device("cuda")
        self.enabled += 1

    def enable_sequential_cpu_offload(self, *a, **kw):
        self.enabled += 1

    def to(self, *a, **kw):
        self._execution_device = torch.device("cuda")
        return self

    def set_adapters(self, names, weights):
        self.adapters = (list(names), list(weights))

    def unload_lora_weights(self):
        self.cleared += 1

    def delete_adapters(self, *a, **kw):
        pass


def _lora(path, hidden=4096, rank=4):
    """LoRA minuscule au dialecte PEFT, a la dimension cachee du 9B."""
    save_file({
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(rank, hidden),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(hidden, rank),
    }, path)
    return path


class _Base9B:
    """Fait croire a la garde de variante que la base tourne en 9B."""

    def __enter__(self):
        self.key = cz.BASE_REPO
        self.had = self.key in cz._BASE_DIM_CACHE
        self.old = cz._BASE_DIM_CACHE.get(self.key)
        cz._BASE_DIM_CACHE[self.key] = 4096
        return self

    def __exit__(self, *exc):
        if self.had:
            cz._BASE_DIM_CACHE[self.key] = self.old
        else:
            cz._BASE_DIM_CACHE.pop(self.key, None)
        return False


def test_loras_are_loaded_with_real_tensors():
    tmp = tempfile.mkdtemp()
    try:
        p = _lora(os.path.join(tmp, "dora.safetensors"))
        pipe = FakePipe()
        with _Base9B():
            ok, applied = cz._sync_adapters(pipe, [(p, 0.8)], [])
        assert ok and applied == [(p, 0.8)], (ok, applied)
        assert len(pipe.loaded) == 1, pipe.loaded
        _a, kw = pipe.loaded[0]
        assert kw.get("low_cpu_mem_usage") is False, kw
        assert pipe.adapters == (["cz_lora_0"], [0.8]), pipe.adapters
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_failed_load_puts_the_offload_back():
    tmp = tempfile.mkdtemp()
    try:
        p = _lora(os.path.join(tmp, "dora.safetensors"))
        pipe = FakePipe(fail=True)
        real_dev, real_off = cz.DEVICE, cz._effective_offload
        cz.DEVICE = "cuda"
        cz._effective_offload = lambda *a, **k: "model"
        try:
            with _Base9B():
                ok, applied = cz._sync_adapters(pipe, [(p, 0.8)], [])
        finally:
            cz.DEVICE, cz._effective_offload = real_dev, real_off
        assert (ok, applied) == (False, []), (ok, applied)
        # Sans la remise en etat, le pipe resterait sur le CPU et tout echouerait apres.
        assert pipe._all_hooks and str(pipe._execution_device) == "cuda", pipe._all_hooks
        assert pipe.enabled == 1, pipe.enabled
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_restore_offload_only_acts_on_a_pipe_left_on_the_cpu():
    real_dev, real_off = cz.DEVICE, cz._effective_offload
    cz.DEVICE = "cuda"
    cz._effective_offload = lambda *a, **k: "model"
    try:
        broken = FakePipe(hooks=False)               # laisse sur le CPU
        assert cz.restore_offload(broken, "a test") is True
        assert broken.enabled == 1 and str(broken._execution_device) == "cuda"
        healthy = FakePipe()                          # deja sur la carte
        assert cz.restore_offload(healthy) is False
        assert healthy.enabled == 0
        cz._effective_offload = lambda *a, **k: "none"
        plain = FakePipe(hooks=False)
        assert cz.restore_offload(plain) is True      # offload 'none' -> simple .to(cuda)
        assert plain.enabled == 0 and str(plain._execution_device) == "cuda"
        cz.DEVICE = "cpu"                             # machine sans carte: on ne touche a rien
        assert cz.restore_offload(FakePipe(hooks=False)) is False
    finally:
        cz.DEVICE, cz._effective_offload = real_dev, real_off


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} LoRA offload tests passed.")
