"""A bench: every model of the library, three images, timings measured.

What it separates -- and that is the whole point, because those three numbers do not
have the same cause and are not fixed the same way:

  LOADING      the pipeline ready to generate, from an empty state (free_vram before
               every model, otherwise we would be measuring a warm transformer swap, not
               a load). That is where the disk read, the FP8/INT8 dequantisation -- or
               its cache, see rebuild_cache.bat -- and the setting up of the offload get
               paid for.
  1st IMAGE    the first generation AFTER loading. Always slower: CUDA kernels to
               compile, an empty embeddings cache, weights moving onto the GPU.
               Confusing it with the steady state makes one conclude "this model is
               slow" over a cost paid only once (seen in this house: 616 s announced
               for a model that ran at 9.6 s on the next go).
  STEADY STATE the average of the images that follow. The only figure that counts for
               real work, and the only one comparable between models.

Both variants are covered. A 4B checkpoint does not load on a 9B base: so the models are
grouped by variant and the base repo is switched once per group (a base change is a full
reload, we do not pay one per file).

The steps come from EVERY MODEL'S PROFILE (the CivitAI consensus, the undistilled flag,
then the file name -- the logic of the selection in the UI), not from a single value:
what is of interest is the time a model takes to render a usable image, not a time per
step artificially levelled. The cost PER STEP is reported next to it, for whoever wants
the other reading. --steps N forces the same value everywhere when the raw comparison is
wanted.

Usage:
    .venv/Scripts/python tools/bench_models.py --list
    .venv/Scripts/python tools/bench_models.py
    (or double-click on bench_models.bat)

  --list          shows the plan, generates nothing
  --only SUBSTR   filters on the file name (repeatable, case-insensitive)
  --size WxH      the resolution (1024x1024 by default). The SAME for all: the model
                  must be the only variable.
  --seed N        the seed (12345 by default), identical everywhere -> comparable images
  --steps N       forces the same number of steps for all
  --resume        skips the models already in bench/results.json

RESUMING: every model is written into bench/results.json as soon as it is finished, and
the report is rewritten right after. An interruption only loses the model in progress.

THE GPU MUST BE FREE. This bench loads a complete model per entry; an instance of the app
running alongside will at best slow the measurement down, at worst overflow the VRAM.

"""
import json
import os
import statistics
import sys
import time
import traceback

USAGE = """Usage: bench_models.py [--list] [--only SUBSTR ...] [--size WxH]
                      [--seed N] [--steps N] [--resume]
Any other option (-h, --help, a typo) prints this and exits: the tool never
starts a multi-hour benchmark by accident."""

ONLY, SIZE, SEED, FORCE_STEPS = [], (1024, 1024), 12345, None
LIST_ONLY = RESUME = False
_args = sys.argv[1:]
_i = 0
while _i < len(_args):
    a = _args[_i]
    if a == "--list":
        LIST_ONLY = True
    elif a == "--resume":
        RESUME = True
    elif a in ("--only", "--size", "--seed", "--steps"):
        if _i + 1 >= len(_args):
            print(USAGE)
            sys.exit(2)
        v = _args[_i + 1]
        try:
            if a == "--only":
                ONLY.append(v.lower())
            elif a == "--size":
                w, h = v.lower().split("x")
                SIZE = (int(w), int(h))
            elif a == "--seed":
                SEED = int(v)
            else:
                FORCE_STEPS = int(v)
        except Exception:
            print(USAGE)
            sys.exit(2)
        _i += 2
        continue
    else:
        print(USAGE)
        sys.exit(0 if a in ("-h", "--help") else 2)
    _i += 1

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import cz_pipeline as P            # noqa: E402
import cz_ui as U                  # noqa: E402
import torch                       # noqa: E402

OUT = os.path.join(HERE, "bench")
RESULTS = os.path.join(OUT, "results.json")
REPORT = os.path.join(OUT, "REPORT.md")

# Three frankly different images: a face (the skin and the eyes are what betrays a
# broken merge first), a wide scene (composition, depth), and TEXT (what FLUX.2 can do
# and what a failed fusion destroys before anything else).
PROMPTS = [
    ("portrait",
     "portrait of a young woman with long dark hair, hoop earrings, scarf, "
     "black and white illustration, soft lighting, clean background, detailed "
     "shading, realistic style, high quality"),
    ("scene",
     "a narrow rainy street in an old european town at dusk, wet cobblestones "
     "reflecting warm shop windows, bicycles leaning on a wall, low fog, "
     "cinematic wide shot, detailed"),
    ("text",
     "a hand-lettered chalkboard sign outside a cafe that reads \"CRISPZ COFFEE - "
     "OPEN TILL LATE\", warm interior light behind it, shallow depth of field, "
     "photographic"),
]


def _variant(path):
    """'4B' / '9B' / None for a file; for a repo, what its config says.

    The GGUF has its own reader: without it a 4B GGUF came out with no variant, so
    paired with the current base -- and a 4B tested on a 9B base always fails."""
    if os.path.isfile(path):
        dim = (P._gguf_hidden_dim(path) if P._is_gguf_path(path)
               else P._flux2_hidden_dim(path))
        return P._FLUX2_VARIANTS.get(dim)
    return P._FLUX2_VARIANTS.get(P._base_hidden_dim(path))


def _plan():
    """[(label, path_or_repo, variant, reason_for_refusal)] -- the refusals are kept in
    the plan to be DISPLAYED, not silently absent."""
    items = []
    for repo in getattr(U, "ZIMAGE_BASE_REPOS", []) or []:
        items.append((os.path.basename(repo), repo, _variant(repo), None))
    seen = set()
    for d in P._checkpoint_dirs():
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f in seen or not f.lower().endswith(
                    (".safetensors", ".ckpt", ".pt", ".sft", ".gguf")):
                continue
            seen.add(f)
            p = os.path.join(d, f)
            var = _variant(p)
            why = None
            if f.lower().endswith(".safetensors"):
                # A VARIANT refusal does not count: we will switch the base for it.
                why = P._safetensors_unsupported(p)
                if why and why == P._flux2_variant_mismatch(P._flux2_hidden_dim(p)):
                    why = None
            elif f.lower().endswith(".gguf"):
                # The same rule for the GGUF. The first version only checked the layout
                # of the .safetensors: two GGUFs converted by stable-diffusion.cpp were
                # ATTEMPTED, then refused at load time by the app's guard. We say so from
                # the plan on, as for any other file the app refuses.
                why = P._gguf_layout_unsupported(p) or None
                # ... but as for the .safetensors, a VARIANT refusal does not count:
                # the bench switches the base per group. Without that line, a 4B base
                # discarded the three 9B GGUFs from the plan -- checked, 22 models instead
                # of 25.
                if why and why == P._flux2_variant_mismatch(P._gguf_hidden_dim(p)):
                    why = None
            items.append((f, p, var, why))
    if ONLY:
        items = [it for it in items if any(s in it[0].lower() for s in ONLY)]
    # Grouped by variant -> a single base repo switch per group.
    order = {"4B": 0, "9B": 1, None: 2}
    items.sort(key=lambda it: (order.get(it[2], 3), it[0].lower()))
    return items


def _base_for(variant):
    """The base repo to set in order to test this variant."""
    for repo in getattr(U, "ZIMAGE_BASE_REPOS", []) or []:
        if _variant(repo) == variant:
            return repo
    return None


def _dequant_state(path):
    """'warm' / 'cold' / '-' : the state of the dequantisation cache BEFORE the test.

    Without that column the LOADING line cannot be interpreted: the same file takes a
    few seconds with its bf16 in the cache and several MINUTES without. Comparing a
    cached model with one that is not does not measure the models."""
    if not os.path.isfile(path) or not str(path).lower().endswith(".safetensors"):
        return "-"
    try:
        if not P._safetensors_dequant(path):
            return "-"                      # bf16: nothing to dequantise
        c = P._dequant_cache_path(path)
        return "chaud" if c and os.path.isfile(c) else "froid"
    except Exception:
        return "-"


def _profile(path):
    """(steps, guidance, source) for this model -- the same logic as the selection in the
    UI, so that the bench measures what the user will really get."""
    if FORCE_STEPS:
        return FORCE_STEPS, 1.0, f"forced (--steps {FORCE_STEPS})"
    if not os.path.isfile(path):
        return 4, 1.0, "base repo default"
    try:
        st, g, why = U._profile_for_checkpoint(path)
        src = ("CivitAI consensus" if "consensus" in (why or "").lower()
               else "undistilled flag" if "undistil" in (why or "").lower()
               else "file name")
        return int(st), float(g), src
    except Exception:
        return 4, 1.0, "fallback"


def _load(prev):
    """Loads the list of the results already obtained."""
    if prev and os.path.isfile(RESULTS):
        try:
            return json.load(open(RESULTS, encoding="utf-8"))
        except Exception:
            pass
    return []


def _write_report(rows):
    ok = [r for r in rows if not r.get("error")]
    ok.sort(key=lambda r: (r.get("warm_s") is None, r.get("warm_s") or 0))
    w, h = SIZE
    lines = [
        "# Banc d'essai des modeles",
        "",
        f"{w}x{h}, seed {SEED}, {len(PROMPTS)} images par modele "
        f"({', '.join(n for n, _ in PROMPTS)}).",
        "",
        "- **Chargement**: pipeline pret, depuis un etat vide (`free_vram` avant chaque",
        "  modele). Lecture disque + dequantification + offload.",
        "- **1re image**: la premiere apres chargement -- noyaux CUDA, cache d'embeddings",
        "  vide. Cout paye une fois, a ne pas confondre avec le regime.",
        "- **Regime**: moyenne des images suivantes. Le seul chiffre comparable.",
        "- **Modeles nus**: aucune LoRA ni LoKr, quelle que soit la config du moment.",
        "- **Jusqu'a la 1re image** = chargement + 1re image. A LIRE EN PREMIER: sous",
        "  offload les poids restent mappes sur le disque et la lecture glisse du",
        "  chargement dans la 1re image (mesure: 4 s + 173 s pour un 9B bf16 sur disque",
        "  USB). Chacune de ces deux colonnes, seule, trompe; leur somme non.",
        "- **s/step** = regime / steps. Surestime le cout d'un step sous offload: une",
        "  part fixe par image (~6,5 s sur le 9B, transferts + VAE) y est repartie.",
        "",
        "- **Cache**: etat du cache de dequantification AVANT le test. `froid` = le",
        "  chargement inclut la conversion FP8/INT8 vers bf16 (minutes). Une colonne",
        "  `chargement` ne se compare qu'a cache egal -- `rebuild_cache.bat` les met",
        "  tous a chaud.",
        "",
        "> Le cache fichier de l'OS pese aussi sur la colonne `chargement`: le meme",
        "> modele, mesure deux fois de suite, est passe de 24.1 s a 10.3 s sans qu'on",
        "> touche a rien. Le premier modele de la liste paie donc un disque froid que",
        "> les suivants ne paient pas. Lire cette colonne comme un ordre de grandeur,",
        "> pas au dixieme de seconde -- `1re image` et `regime`, eux, sont fiables.",
        "",
        "| Modele | Var. | Steps (source) | Cache | Chargement | 1re image | Jusqu'a la 1re | Regime | s/step | VRAM max |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ok:
        warm = r.get("warm_s")
        per = f"{warm / r['steps']:.2f}" if warm and r.get("steps") else "-"
        lines.append(
            f"| `{r['name']}` | {r.get('variant') or '?'} | {r['steps']} "
            f"({r['profile_source']}) | {r.get('dequant_cache', '-')} | "
            f"{r['load_s']:.1f} s | {r['first_s']:.1f} s | "
            f"**{r['load_s'] + r['first_s']:.0f} s** | "
            + (f"{warm:.1f} s" if warm else "-")
            + f" | {per} | {r.get('vram_peak_gb', 0):.1f} Go |")
    bad = [r for r in rows if r.get("error")]
    if bad:
        lines += ["", "## Non testes", "",
                  "| Modele | Raison |", "|---|---|"]
        for r in bad:
            lines.append(f"| `{r['name']}` | {str(r['error'])[:160]} |")
    lines += ["", f"_Genere le {time.strftime('%Y-%m-%d %H:%M')} par "
                  f"`tools/bench_models.py`._", ""]
    os.makedirs(OUT, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))


def _bench_one(name, path, variant):
    """Measures a model. Returns the result row (with 'error' on a failure)."""
    steps, guidance, src = _profile(path)
    row = {"name": name, "variant": variant, "steps": steps, "guidance": guidance,
           "profile_source": src, "dequant_cache": _dequant_state(path)}
    # The base of the right variant first (a full reload), then the transformer.
    base = _base_for(variant) or P.BASE_REPO
    if base != P.BASE_REPO:
        print(f"    base repo -> {base}")
        P.set_zimage_model(base)
    P.set_zimage_transformer(path if os.path.isfile(path) else None)
    # The model ALONE: no LoRA, no LoKr, no edit set. LORAS is initialised from
    # `default_loras` when the app starts; without that reset the bench would measure the
    # config of the day and not the model -- a merged LoKr adds ~24 s to every load,
    # changes every image, and a 9B set on a 4B base would pour its warnings in there.
    P.LORAS = []
    P.EDIT_LORAS = []
    # An empty state BEFORE the measurement: otherwise we would be measuring a warm
    # transformer swap (the VAE + the encoder kept), not a load.
    P.free_vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    P.set_guidance(guidance)

    t0 = time.time()
    P.get_pipe("txt2img")
    row["load_s"] = time.time() - t0
    print(f"    chargement {row['load_s']:.1f} s")

    times, outdir = [], os.path.join(OUT, name.replace(os.sep, "_"))
    os.makedirs(outdir, exist_ok=True)
    for i, (label, prompt) in enumerate(PROMPTS):
        t = time.time()
        img = P.generate(prompt, SIZE[0], SIZE[1], steps, SEED)
        dt = time.time() - t
        times.append(dt)
        if isinstance(img, (list, tuple)):
            img = img[0]
        try:
            img.save(os.path.join(outdir, f"{i + 1}_{label}.png"))
        except Exception as e:
            print(f"    (image {label} non sauvee: {e})")
        print(f"    {label}: {dt:.1f} s" + ("  <- 1re" if i == 0 else ""))
    row["first_s"] = times[0]
    row["warm_s"] = statistics.fmean(times[1:]) if len(times) > 1 else None
    row["times_s"] = times
    if torch.cuda.is_available():
        row["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    return row


def main():
    items = _plan()
    done = {r["name"] for r in _load(RESUME)}
    todo = [it for it in items if it[3] is None and it[0] not in done]
    skipped = [it for it in items if it[3]]

    print(f"resolution {SIZE[0]}x{SIZE[1]} | seed {SEED} | "
          f"{len(PROMPTS)} images par modele")
    for n, p, var, why in items:
        mark = ("DEJA FAIT " if n in done else
                "REFUSE     " if why else "A TESTER   ")
        st, _g, src = (FORCE_STEPS, 1.0, "forced") if FORCE_STEPS else _profile(p)
        dq = _dequant_state(p)
        extra = why or (f"{var or '?'}, {st} steps ({src})"
                        + (f", dequant {dq}" if dq != "-" else ""))
        # With no variant signature we do not know which base to pair it with: it goes
        # LAST (the sort puts it at the end) and we say so, rather than throwing it away.
        if not why and var is None:
            extra += "  [pas de signature de variante -> base courante, peut echouer]"
        print(f"{mark} {n[:44]:46s} {extra[:96]}")
    print(f"\n{len(todo)} modele(s) a tester, {len(skipped)} refuse(s), "
          f"{len(done)} deja fait(s).")
    if skipped:
        print("Les refuses ne sont pas des echecs du banc: ce sont des fichiers que "
              "l'app ne charge pas (LoRA egaree, FP4, LyCORIS...), avec leur raison.")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {free / 1024**3:.1f} Go libres sur {total / 1024**3:.1f}. "
              "Ferme l'app avant de lancer, sinon la mesure ne vaut rien.")
    if LIST_ONLY or not todo:
        return

    rows = _load(RESUME)
    for i, (name, path, var, _why) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {name}")
        try:
            rows.append(_bench_one(name, path, var))
        except KeyboardInterrupt:
            print("interrompu")
            break
        except Exception as e:
            print(f"    ECHEC {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            rows.append({"name": name, "variant": var, "steps": 0, "guidance": 0,
                         "profile_source": "-", "error": f"{type(e).__name__}: {e}"})
        # Written AFTER EVERY model: an interruption only loses the one in progress.
        os.makedirs(OUT, exist_ok=True)
        with open(RESULTS, "w", encoding="utf-8", newline="") as f:
            json.dump(rows, f, indent=1)
        _write_report(rows)

    _write_report(rows)
    print(f"\nRapport: {REPORT}\nImages : {OUT}")


if __name__ == "__main__":
    main()
