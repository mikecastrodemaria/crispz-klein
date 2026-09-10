"""Pre-remplit le cache de dequantification (cache/dequant) pour TOUS les checkpoints
FP8/INT8 des dossiers de modeles, pour ne pas payer la conversion a la premiere
utilisation (elle bloque alors l'UI plusieurs minutes en plein travail).

Usage:
    .venv/Scripts/python tools/rebuild_dequant_cache.py [--list] [--cpu] [--only SUBSTR]
    (ou double-clic sur rebuild_cache.bat a la racine)

- REPRISE GRATUITE: un checkpoint deja en cache est saute en une seconde -> relancable
  a volonte, y compris apres une coupure.
- --list : montre ce qui serait fait, sans rien convertir.
- --cpu  : dequantification sans toucher au GPU (par defaut: GPU si present, cf.
  convert_device). A preferer si un rendu tourne en meme temps.

Les DEUX variantes sont pre-remplies, 4B comme 9B, quelle que soit celle du repo de
base courant: la dequantification ne depend que du fichier (la clef de cache est
chemin+taille+mtime), et le but est justement que basculer de base soit instantane.
Seul le CHARGEMENT exige que la variante corresponde au repo choisi.

Ne concerne QUE les .safetensors FP8/INT8:
  - .gguf          -> reste quantifie en VRAM, aucune dequantification a cacher;
  - bf16/fp16      -> rien a dequantifier (un cache serait une copie bf16 -> bf16),
                      y compris au layout ComfyUI ou seul le prefixe est retire;
  - LoRA/SVDQuant  -> non chargeables, ignores avec leur raison.

Chaque entree pese le poids du build BF16, MESURE sur l'en-tete du fichier et non
estime: ~16.9 Go pour un transformer klein-9B, ~7.2 Go pour un 4B, davantage pour un
bundle qui embarque son encodeur texte. Le total est verifie contre
dequant_cache_max_gb (config.txt) ET contre la place libre du disque, sinon les
premieres conversions seraient evincees par les dernieres et le cache ne servirait a
rien. Supprimer cache/dequant est toujours sur (il se reconstruit a la demande).
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

# Tenseurs qui ne survivent PAS au dequant: les facteurs d'echelle et le descripteur
# comfy_quant. Les compter gonflerait l'estimation d'un build a scales par ligne.
_SCALE_SUFFIXES = ("_scale", "_scale_inv", ".scale_weight", ".comfy_quant")


def bf16_gb(path):
    """Poids du bf16 qui sera ecrit dans le cache, lu a l'EN-TETE seule (aucun
    chargement). Recoupe avec le cache existant: 16.9 Go annonces, 16.9 Go sur le
    disque pour rayKlein9bBFS_fp8V2.

    Bundle tout-en-un (transformer + encodeur texte + VAE, prefixe ComfyUI): on ne
    compte QUE le transformer, exactement le filtre de _load_dequant_state_dict. La
    premiere version comptait le fichier entier et annoncait 29.4 Go pour
    gonzalomoKlein_v10, qui en ecrit 16.9 -- 936 tenseurs d'un Qwen3-8B et le VAE ne
    vont jamais dans le cache. Erreur dans le sens prudent, mais le plafond reclame
    etait gonfle d'autant."""
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
        # Variante differente du repo de base COURANT: ca n'empeche pas de pre-remplir
        # son cache, seulement de la charger maintenant. On la convertit quand meme --
        # sinon basculer 4B <-> 9B repaierait la conversion, ce qui est exactement ce
        # que ce script existe pour eviter.
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
