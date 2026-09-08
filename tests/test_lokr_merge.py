"""LoKr LyCORIS: fusion dans les poids (dW = w1 (x) w2).

Pourquoi une fusion et pas un adaptateur: sur FLUX.2 le q/k/v est FUSIONNE cote
checkpoint ([3d, d]) et SEPARE cote diffusers (trois [d, d]). Un produit de Kronecker
ne se tranche pas en trois -- sur SNOFS, w1 est [4,4] et w2 [3072,1024], donc les blocs
de w1 font 3072 lignes la ou la coupe en tombe sur 4096. Le delta MATERIALISE, lui, se
coupe comme n'importe quelle matrice. C'est ce que ces tests verrouillent, en passant
par le convertisseur de cles de diffusers lui-meme (le meme que from_single_file), pas
par une table de correspondance recopiee a la main qui pourrait deriver de lui.

L'echelle est l'autre piege. LyCORIS n'applique AUCUN scalaire quand w1 et w2 sont
pleines (il n'y a pas de rang), et peft calcule alpha/r. Les fichiers ai-toolkit
ecrivent alors alpha = lora_dim -- 1e10 sur SNOFS, mesure sur le fichier reel -- si
bien que les deux conventions donnent 1.0. Les deux sont testees.

Run:  .venv/Scripts/python tests/test_lokr_merge.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_lokr_merge")
os.makedirs(TMP, exist_ok=True)


class _FakeTransformer:
    """Juste ce que _merge_lokr consomme: named_parameters()."""

    def __init__(self, shapes):
        self._p = {k: torch.nn.Parameter(torch.zeros(*s)) for k, s in shapes.items()}

    def named_parameters(self):
        return list(self._p.items())

    def __getitem__(self, k):
        return self._p[k].data


def _write(name, tensors):
    p = os.path.join(TMP, name)
    save_file(tensors, p)
    return p


def test_the_kronecker_product_lands_on_the_right_weight():
    """Couche simple: double_blocks.0.img_attn.proj -> attn.to_out.0."""
    w1 = torch.randn(2, 2)
    w2 = torch.randn(2, 2)
    p = _write("lokr_proj.safetensors", {
        "diffusion_model.double_blocks.0.img_attn.proj.alpha": torch.tensor(1e10),
        "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1": w1,
        "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2": w2,
    })
    t = _FakeTransformer({"transformer_blocks.0.attn.to_out.0.weight": (4, 4)})
    n, problems = P._merge_lokr(t, p, 1.0)
    assert (n, problems) == (1, []), (n, problems)
    got = t["transformer_blocks.0.attn.to_out.0.weight"]
    assert torch.allclose(got, torch.kron(w1, w2), atol=1e-5), (got, torch.kron(w1, w2))
    print("OK test_the_kronecker_product_lands_on_the_right_weight")


def test_a_fused_qkv_is_split_into_three_weights():
    """Le cas qui interdit l'approche adaptateur: [3d, d] -> to_q / to_k / to_v.
    On coupe le delta, jamais les facteurs -- et la coupe ne tombe volontairement PAS
    sur un bloc de w1 (w2 a 3 lignes, la coupe en fait 2), comme sur SNOFS."""
    w1 = torch.randn(2, 2)
    w2 = torch.randn(3, 4)                     # -> delta [6, 8], chunk 3 -> [2, 8]
    p = _write("lokr_qkv.safetensors", {
        "diffusion_model.double_blocks.0.img_attn.qkv.alpha": torch.tensor(1e10),
        "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1": w1,
        "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2": w2,
    })
    keys = [f"transformer_blocks.0.attn.to_{x}.weight" for x in ("q", "k", "v")]
    t = _FakeTransformer({k: (2, 8) for k in keys})
    n, problems = P._merge_lokr(t, p, 1.0)
    assert (n, problems) == (3, []), (n, problems)
    expect = torch.chunk(torch.kron(w1, w2), 3, dim=0)
    for k, e in zip(keys, expect):
        assert torch.allclose(t[k], e, atol=1e-5), k
    # et le recollage des trois redonne bien le kron entier
    assert torch.allclose(torch.cat([t[k] for k in keys], dim=0),
                          torch.kron(w1, w2), atol=1e-5)
    print("OK test_a_fused_qkv_is_split_into_three_weights")


def test_the_lora_weight_scales_the_delta():
    w1, w2 = torch.randn(2, 2), torch.randn(2, 2)
    p = _write("lokr_w.safetensors", {
        "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1": w1,
        "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2": w2,
    })
    t = _FakeTransformer({"transformer_blocks.0.attn.to_out.0.weight": (4, 4)})
    P._merge_lokr(t, p, 0.5)
    assert torch.allclose(t["transformer_blocks.0.attn.to_out.0.weight"],
                          0.5 * torch.kron(w1, w2), atol=1e-5)
    print("OK test_the_lora_weight_scales_the_delta")


def test_full_factors_use_no_scalar():
    """SNOFS: w1 et w2 pleines, alpha = 1e10 (lora_dim sentinelle). Aucun scalaire."""
    mod = {"lokr_w1": torch.randn(2, 2), "lokr_w2": torch.randn(2, 2),
           "alpha": torch.tensor(1e10)}
    assert P._lokr_scale(mod, None) == 1.0
    d = P._lokr_delta(mod)
    assert torch.allclose(d, torch.kron(mod["lokr_w1"], mod["lokr_w2"]), atol=1e-5)
    print("OK test_full_factors_use_no_scalar")


def test_factored_factors_use_alpha_over_rank():
    """Forme factorisee: w1 = w1_a @ w1_b, rang 2, alpha 8 -> echelle 4, comme peft."""
    a, b = torch.randn(4, 2), torch.randn(2, 4)
    mod = {"lokr_w1_a": a, "lokr_w1_b": b, "lokr_w2": torch.randn(2, 2),
           "alpha": torch.tensor(8.0)}
    assert P._lokr_scale(mod, 2) == 4.0
    d = P._lokr_delta(mod)
    assert torch.allclose(d, torch.kron(a @ b, mod["lokr_w2"]) * 4.0, atol=1e-4)
    print("OK test_factored_factors_use_alpha_over_rank")


def test_a_delta_without_a_target_is_reported_not_dropped():
    """La regle de la maison: rien ne disparait en silence. Un module que le modele
    n'a pas doit REMONTER, sinon un mapping qui derive donnerait un merge a moitie
    vide et un rendu presque normal -- le pire des cas."""
    p = _write("lokr_orphan.safetensors", {
        "diffusion_model.double_blocks.7.img_attn.proj.lokr_w1": torch.randn(2, 2),
        "diffusion_model.double_blocks.7.img_attn.proj.lokr_w2": torch.randn(2, 2),
    })
    t = _FakeTransformer({"transformer_blocks.0.attn.to_out.0.weight": (4, 4)})
    n, problems = P._merge_lokr(t, p, 1.0)
    assert n == 0, n
    assert len(problems) == 1 and "no such weight" in problems[0], problems
    print("OK test_a_delta_without_a_target_is_reported_not_dropped")


def test_a_shape_mismatch_is_reported_not_applied():
    p = _write("lokr_badshape.safetensors", {
        "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1": torch.randn(2, 2),
        "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2": torch.randn(2, 2),
    })
    t = _FakeTransformer({"transformer_blocks.0.attn.to_out.0.weight": (8, 8)})
    n, problems = P._merge_lokr(t, p, 1.0)
    assert n == 0 and len(problems) == 1 and "vs weight" in problems[0], problems
    assert torch.count_nonzero(t["transformer_blocks.0.attn.to_out.0.weight"]) == 0
    print("OK test_a_shape_mismatch_is_reported_not_applied")


def test_a_lokr_is_kept_out_of_the_peft_set():
    """Ce qui part chez peft ne doit plus contenir la LoKr (elle est dans les poids),
    mais doit encore contenir le LoHa, pour que _sync_adapters le refuse par son nom."""
    lokr = _write("set_lokr.safetensors", {
        f"diffusion_model.double_blocks.{i}.img_attn.proj.lokr_w{j}": torch.zeros(2, 2)
        for i in range(3) for j in (1, 2)})
    loha = _write("set_loha.safetensors", {
        f"diffusion_model.double_blocks.{i}.img_attn.proj.hada_w{j}_a": torch.zeros(2, 2)
        for i in range(3) for j in (1, 2)})
    lora = _write("set_lora.safetensors", {
        f"transformer.blocks.{i}.attn.to_q.lora_{ab}.weight": torch.zeros(2, 2)
        for i in range(3) for ab in ("A", "B")})
    s = [(lokr, 1.0), (loha, 1.0), (lora, 1.0)]
    assert P._lokr_set(s) == [(lokr, 1.0)]
    assert P._peft_set(s) == [(loha, 1.0), (lora, 1.0)]
    assert P._lora_unsupported(lokr) is None          # supporte, par fusion
    assert "LoHa" in (P._lora_unsupported(loha) or "")
    assert P._lora_unsupported(lora) is None
    print("OK test_a_lokr_is_kept_out_of_the_peft_set")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All LoKr merge tests passed.")
