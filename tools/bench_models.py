"""Banc d'essai: chaque modele de la bibliotheque, trois images, temps mesures.

Ce qu'il separe -- et c'est tout l'interet, parce que ces trois nombres n'ont pas la
meme cause et ne se corrigent pas pareil:

  CHARGEMENT   pipeline pret a generer, depuis un etat vide (free_vram avant chaque
               modele, sinon on mesurerait un echange de transformer a chaud, pas un
               chargement). C'est la que se paient la lecture disque, la
               dequantification FP8/INT8 -- ou son cache, cf. rebuild_cache.bat -- et
               la mise en place de l'offload.
  1re IMAGE    la premiere generation APRES chargement. Toujours plus lente: noyaux
               CUDA a compiler, cache d'embeddings vide, poids qui montent sur le GPU.
               La confondre avec le regime etabli fait conclure "ce modele est lent"
               sur un cout paye une seule fois (vu dans cette maison: 616 s annonces
               pour un modele qui tournait a 9.6 s au coup suivant).
  REGIME       la moyenne des images suivantes. Le seul chiffre qui vaut pour un
               travail reel, et le seul comparable entre modeles.

Les deux variantes sont couvertes. Un checkpoint 4B ne se charge pas sur une base 9B:
les modeles sont donc groupes par variante et le repo de base est bascule une fois par
groupe (un changement de base est un rechargement complet, on n'en paie pas un par
fichier).

Les steps viennent du PROFIL DE CHAQUE MODELE (consensus CivitAI, drapeau undistilled,
puis nom de fichier -- la logique de la selection dans l'UI), pas d'une valeur unique:
c'est le temps qu'un modele met a rendre une image utilisable qui interesse, pas un
temps par step artificiellement egalise. Le cout PAR STEP est reporte a cote, pour qui
veut l'autre lecture. --steps N force la meme valeur partout si on veut la comparaison
brute.

Usage:
    .venv/Scripts/python tools/bench_models.py --list
    .venv/Scripts/python tools/bench_models.py
    (ou double-clic sur bench_models.bat)

  --list          montre le plan, ne genere rien
  --only SUBSTR   filtre sur le nom de fichier (repetable, insensible a la casse)
  --size WxH      resolution (defaut 1024x1024). La MEME pour tous: le modele doit
                  etre la seule variable.
  --seed N        graine (defaut 12345), identique partout -> images comparables
  --steps N       force le meme nombre de steps pour tous
  --resume        saute les modeles deja dans bench/results.json

REPRISE: chaque modele est ecrit dans bench/results.json des qu'il est fini, et le
rapport est reecrit dans la foulee. Une coupure ne perd que le modele en cours.

LE GPU DOIT ETRE LIBRE. Ce banc charge un modele complet par entree; une instance de
l'app qui tourne a cote fera au mieux ralentir la mesure, au pire deborder la VRAM.
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

# Trois images franchement differentes: un visage (la peau et les yeux sont ce qui
# trahit un merge casse en premier), une scene large (composition, profondeur), et du
# TEXTE (ce que FLUX.2 sait faire et qu'une fusion ratee detruit avant tout le reste).
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
    """'4B' / '9B' / None pour un fichier; pour un repo, ce que dit sa config.

    Le GGUF a son propre lecteur: sans lui un GGUF 4B ressortait sans variante, donc
    apparie a la base courante -- et un 4B teste sur une base 9B echoue toujours."""
    if os.path.isfile(path):
        dim = (P._gguf_hidden_dim(path) if P._is_gguf_path(path)
               else P._flux2_hidden_dim(path))
        return P._FLUX2_VARIANTS.get(dim)
    return P._FLUX2_VARIANTS.get(P._base_hidden_dim(path))


def _plan():
    """[(etiquette, chemin_ou_repo, variante, raison_de_refus)] -- les refus sont
    gardes dans le plan pour etre AFFICHES, pas silencieusement absents."""
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
                # Le refus de VARIANTE ne compte pas: on basculera la base pour lui.
                why = P._safetensors_unsupported(p)
                if why and why == P._flux2_variant_mismatch(P._flux2_hidden_dim(p)):
                    why = None
            elif f.lower().endswith(".gguf"):
                # Meme regle pour le GGUF. La premiere version ne verifiait le layout
                # que des .safetensors: deux GGUF convertis par stable-diffusion.cpp
                # ont ete TENTES, puis refuses au chargement par la garde de l'app. On
                # le dit des le plan, comme pour tout autre fichier que l'app refuse.
                why = P._gguf_layout_unsupported(p) or None
                # ... mais comme pour les .safetensors, le refus de VARIANTE ne compte
                # pas: le banc bascule la base par groupe. Sans cette ligne, une base
                # en 4B ecartait les trois GGUF 9B du plan -- verifie, 22 modeles au
                # lieu de 25.
                if why and why == P._flux2_variant_mismatch(P._gguf_hidden_dim(p)):
                    why = None
            items.append((f, p, var, why))
    if ONLY:
        items = [it for it in items if any(s in it[0].lower() for s in ONLY)]
    # Groupe par variante -> une seule bascule de repo de base par groupe.
    order = {"4B": 0, "9B": 1, None: 2}
    items.sort(key=lambda it: (order.get(it[2], 3), it[0].lower()))
    return items


def _base_for(variant):
    """Le repo de base a poser pour tester cette variante."""
    for repo in getattr(U, "ZIMAGE_BASE_REPOS", []) or []:
        if _variant(repo) == variant:
            return repo
    return None


def _dequant_state(path):
    """'chaud' / 'froid' / '-' : l'etat du cache de dequantification AVANT le test.

    Sans cette colonne la ligne CHARGEMENT est ininterpretable: le meme fichier met
    quelques secondes avec son bf16 en cache et plusieurs MINUTES sans. Comparer un
    modele cache a un modele qui ne l'est pas ne mesure pas les modeles."""
    if not os.path.isfile(path) or not str(path).lower().endswith(".safetensors"):
        return "-"
    try:
        if not P._safetensors_dequant(path):
            return "-"                      # bf16: rien a dequantifier
        c = P._dequant_cache_path(path)
        return "chaud" if c and os.path.isfile(c) else "froid"
    except Exception:
        return "-"


def _profile(path):
    """(steps, guidance, source) pour ce modele -- la meme logique que la selection
    dans l'UI, pour que le banc mesure ce que l'utilisateur obtiendra vraiment."""
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
    """Charge la liste des resultats deja obtenus."""
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
    """Mesure un modele. Renvoie la ligne de resultat (avec 'error' si echec)."""
    steps, guidance, src = _profile(path)
    row = {"name": name, "variant": variant, "steps": steps, "guidance": guidance,
           "profile_source": src, "dequant_cache": _dequant_state(path)}
    # Base de la bonne variante d'abord (rechargement complet), puis le transformer.
    base = _base_for(variant) or P.BASE_REPO
    if base != P.BASE_REPO:
        print(f"    base repo -> {base}")
        P.set_zimage_model(base)
    P.set_zimage_transformer(path if os.path.isfile(path) else None)
    # Le modele SEUL: aucune LoRA, aucune LoKr, aucun jeu d'edition. LORAS est
    # initialise depuis `default_loras` au demarrage de l'app; sans cette remise a
    # zero le banc mesurerait la config du jour et non le modele -- une LoKr fusionnee
    # ajoute ~24 s a chaque chargement, change chaque image, et un 9B pose sur une
    # base 4B y deverserait ses avertissements.
    P.LORAS = []
    P.EDIT_LORAS = []
    # Etat vide AVANT la mesure: sinon on mesurerait un echange de transformer a
    # chaud (VAE + encodeur gardes), pas un chargement.
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
        # Sans signature de variante on ne sait pas a quelle base l'apparier: il passe
        # en DERNIER (le tri le place en fin) et on le dit, plutot que de le jeter.
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
        # Ecrit APRES CHAQUE modele: une coupure ne perd que celui en cours.
        os.makedirs(OUT, exist_ok=True)
        with open(RESULTS, "w", encoding="utf-8", newline="") as f:
            json.dump(rows, f, indent=1)
        _write_report(rows)

    _write_report(rows)
    print(f"\nRapport: {REPORT}\nImages : {OUT}")


if __name__ == "__main__":
    main()
