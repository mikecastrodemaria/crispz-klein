"""Pre-fills the dequantisation cache (cache/dequant) for ALL the FP8/INT8 checkpoints
of the model folders, so as not to pay for the conversion on the first use (it then
blocks the UI for several minutes in the middle of the work).

Usage:
    .venv/Scripts/python tools/rebuild_dequant_cache.py [--list] [--cpu] [--only SUBSTR]
    (or double-click on rebuild_cache.bat at the root)

- RESUMING IS FREE: a checkpoint already cached is skipped in a second -> re-runnable at
  will, including after an interruption.
- --list : shows what would be done, without converting anything.
- --cpu  : dequantises without touching the GPU (by default: the GPU when there is one,
  see convert_device). To be preferred when a render is running at the same time.

BOTH variants are pre-filled, the 4B as well as the 9B, whatever the current base repo's
is: the dequantisation depends on the file only (the cache key is path+size+mtime), and
the whole point is precisely that switching base be instant. Only LOADING requires the
variant to match the chosen repo.

It concerns ONLY the FP8/INT8 .safetensors:
  - .gguf          -> stays quantised in VRAM, no dequantisation to cache;
  - bf16/fp16      -> nothing to dequantise (a cache would be a bf16 -> bf16 copy),
                      including in the ComfyUI layout where only the prefix is removed;
  - LoRA/SVDQuant  -> not loadable, skipped with their reason.

Every entry weighs the size of the BF16 build, MEASURED on the file's header and not
estimated: ~16.9 GB for a klein-9B transformer, ~7.2 GB for a 4B, more for a bundle that
carries its text encoder. The total is checked against dequant_cache_max_gb (config.txt)
AND against the disk's free space, otherwise the first conversions would be evicted by
the last ones and the cache would be of no use. Deleting cache/dequant is always safe (it
rebuilds itself on demand).

"""
import os
import sys
import gc
import math
import shutil
import time

USAGE = """Usage: rebuild_dequant_cache.py [--list] [--cpu] [--only SUBSTR ...]
  --list          show what would be converted, convert nothing
  --cpu           dequantize on the CPU (GPU busy with a render)
  --only SUBSTR   only checkpoints whose file name contains SUBSTR (repeatable,
                  case-insensitive), e.g. --only rayKlein --only fp8
Any other option (-h, --help, a typo) prints this and exits: the tool never
starts a multi-hour conversion by accident."""

_KNOWN = {"--list", "--cpu", "--only"}
ONLY = []
_args = sys.argv[1:]
_i = 0
while _i < len(_args):
    a = _args[_i]
    if a == "--only":
        if _i + 1 >= len(_args):
            print(USAGE)
            sys.exit(2)
        ONLY.append(_args[_i + 1].lower())
        _i += 2
        continue
    if a not in _KNOWN:
        print(USAGE)
        sys.exit(0 if a in ("-h", "--help") else 2)
    _i += 1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as czp  # noqa: E402

if "--cpu" in sys.argv:
    czp.CONFIG["convert_device"] = "cpu"

# The tensors that do NOT survive the dequant: the scale factors and the comfy_quant
# descriptor. Counting them would inflate the estimate of a build with per-row scales.
_SCALE_SUFFIXES = ("_scale", "_scale_inv", ".scale_weight", ".comfy_quant")


def bf16_gb(path):
    """The size of the bf16 that will be written into the cache, read from the HEADER
    alone (no loading). Cross-checked with the existing cache: 16.9 GB announced, 16.9 GB
    on the disk for rayKlein9bBFS_fp8V2.

    An all-in-one bundle (transformer + text encoder + VAE, the ComfyUI prefix): we count
    ONLY the transformer, exactly _load_dequant_state_dict's filter. The first version
    counted the whole file and announced 29.4 GB for gonzalomoKlein_v10, which writes
    16.9 -- the 936 tensors of a Qwen3-8B and the VAE never go into the cache. An error on
    the cautious side, but the cap it asked for was inflated by as much."""
    try:
        hdr = czp._safetensors_header(path)
    except Exception:
        return 0.0
    prefix = czp._COMFY_PREFIX if any(
        k.startswith(czp._COMFY_PREFIX) for k in hdr if k != "__metadata__") else ""
    n = 0
    for k, v in hdr.items():
        if k == "__metadata__" or not isinstance(v, dict) or k.endswith(_SCALE_SUFFIXES):
            continue
        if prefix and not k.startswith(prefix):
            continue
        c = 1
        for s in (v.get("shape") or []):
            c *= int(s)
        n += c
    return n * 2 / 1024 ** 3


if czp._dequant_cache_dir() is None:
    print("dequant_cache est sur 'off' dans config.txt: rien a pre-remplir.")
    sys.exit(0)

todo, done, skipped = [], [], []
for d in czp._checkpoint_dirs():
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if not os.path.isfile(p) or not f.lower().endswith(".safetensors"):
            continue
        if ONLY and not any(s in f.lower() for s in ONLY):
            continue
        dim = czp._flux2_hidden_dim(p)
        bad = czp._safetensors_unsupported(p)
        # A variant different from the CURRENT base repo's: that does not prevent
        # pre-filling its cache, only loading it right now. We convert it anyway --
        # otherwise switching 4B <-> 9B would pay for the conversion again, which is
        # exactly what this script exists to avoid.
        if bad and bad == czp._flux2_variant_mismatch(dim):
            bad = None
        if bad:
            skipped.append((f, bad))
            continue
        dq = czp._safetensors_dequant(p)
        if not dq:
            skipped.append((f, "bf16/fp16, rien a dequantifier"))
            continue
        tag = f"{dq}, {czp._variant_name(dim)}" if dim else dq
        cached = czp._dequant_cache_path(p)
        (done if cached and os.path.isfile(cached) else todo).append((p, tag, bf16_gb(p)))

for f, why in skipped:
    print(f"SKIP {f}: {why}")
for p, tag, gb in done:
    print(f"DEJA EN CACHE {os.path.basename(p)} ({tag}, {gb:.1f} Go)")
for p, tag, gb in todo:
    print(f"A CONVERTIR   {os.path.basename(p)} ({tag}, {gb:.1f} Go)")

if not todo and not done:
    print("\nAucun checkpoint FP8/INT8 trouve dans:", czp._checkpoint_dirs(),
          f"(filtre --only {ONLY})" if ONLY else "")
    sys.exit(0)

cap = czp.DEQUANT_CACHE_MAX_GB
todo_gb = sum(g for _p, _t, g in todo)
need = todo_gb + sum(g for _p, _t, g in done)
print(f"\n{len(todo) + len(done)} checkpoint(s) a couvrir: {need:.0f} Go de cache au "
      f"total, dont {todo_gb:.0f} Go a ecrire maintenant. Plafond "
      f"dequant_cache_max_gb = " + ("illimite (0)." if cap <= 0 else f"{cap:.0f} Go."))

blocked = False
if 0 < cap < need:
    blocked = True
    advise = int(math.ceil(need / 10.0) * 10) + 10
    print(f"ATTENTION: plafond {cap:.0f} Go < {need:.0f} Go necessaires -> les "
          f"premieres conversions seraient evincees par les dernieres et le cache ne "
          f"servirait a rien.\nMets \"dequant_cache_max_gb\": {advise} dans "
          f"config.txt (ou 0 pour illimite) avant de continuer.")
try:
    free = shutil.disk_usage(czp._dequant_cache_dir()).free / 1024 ** 3
except OSError:
    free = None
if free is not None and todo_gb and free < todo_gb:
    blocked = True
    print(f"ATTENTION: {free:.0f} Go libres sur le disque du cache pour {todo_gb:.0f} "
          f"Go a ecrire -> la conversion s'arreterait en route. Fais de la place, ou "
          f"pointe \"dequant_cache\" vers un autre disque dans config.txt.")
if blocked and "--list" not in sys.argv:
    sys.exit(1)

if "--list" in sys.argv:
    sys.exit(0)

t_all = time.time()
ok = fail = 0
for i, (p, tag, gb) in enumerate(todo, 1):
    name = os.path.basename(p)
    t0 = time.time()
    print(f"\n[{i}/{len(todo)}] {name} ({tag}, {gb:.1f} Go) ...")
    try:
        sd = czp._load_dequant_state_dict(p)
        czp._dequant_cache_store(p, sd)
        del sd
        gc.collect()
        ok += 1
        print(f"[{i}/{len(todo)}] OK {name} en {(time.time() - t0) / 60:.1f} min")
    except Exception as e:
        fail += 1
        print(f"[{i}/{len(todo)}] FAIL {name}: {type(e).__name__}: {e}")

print(f"\nTermine en {(time.time() - t_all) / 60:.0f} min: {ok} converti(s), "
      f"{len(done)} deja en cache, {fail} echec(s).")
print("Relancable a volonte: tout ce qui est fait est saute.")
