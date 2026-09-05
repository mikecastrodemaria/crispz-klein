"""Registre des LoRA d'EDITION. VIDE chez crispz-klein.

Le catalogue de l'amont (crispz-qwen-edit) liste des LoRA entrainees pour
Qwen-Image-Edit 2509/2511. Elles sont INCOMPATIBLES avec FLUX.2 Klein:
architecture differente, cles differentes, les charger produirait au mieux une
erreur, au pire des poids incoherents. Les annoncer dans `caps.edit_loras`
reviendrait a promettre une capacite qui casse -- exactement ce que la regle
maison interdit (degradation annoncee, jamais silencieuse).

`EDIT_LORA_SPECS` est donc vide et le catalogue amont est conserve juste en
dessous, commente, comme reference de merge: quand des LoRA d'edition FLUX.2
Klein existeront, elles se declarent ici, au meme format.

Toute la mecanique (telechargement paresseux, resolution de chemin, index
local, overrides config) est INCHANGEE et fonctionne des qu'une entree est
ajoutee. `speed_names()` / `resolve_speed()` (presets Lightning) ne renvoient
plus rien non plus: klein est deja distille a 4 steps, il n'y a pas de LoRA
d'acceleration a empiler.

Surcharge possible dans config.txt (meme format que l'amont):
    "edit_loras_dir": "",
    "edit_loras": {"Mon-Preset": {"repo": "...", "weights": "x.safetensors",
                                  "adapter_name": "mon-preset", "prompt": "...",
                                  "inputs": 1}}
"""
import os

from cz_core import CONFIG, _log, _dbg

# Ordre = ordre du dropdown. inputs = nombre d'images attendu (2 = input + reference).
# Ordre = ordre du dropdown. inputs = nombre d'images attendu (2 = input + reference).
# VIDE: aucune LoRA d'edition FLUX.2 Klein publiee a ce jour (2026-09-05).
EDIT_LORA_SPECS = {}

# --- Catalogue Qwen-Image-Edit de l'amont, conserve comme REFERENCE DE MERGE.
# --- Ne PAS le reactiver tel quel: ces poids ne chargent pas sur FLUX.2.
# EDIT_LORA_SPECS = {
#     "Multiple-Angles": {
#         "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
#         "weights": "镜头转换.safetensors",
#         "adapter_name": "multiple-angles",
#         "prompt": "Rotate the camera 45 degrees to the right.",
#         "inputs": 1, "base": "2509",
#         # meme fichier tel que Civitai le nomme (bibliotheques existantes)
#         "local_names": ["Qwen-Edit-2509-Multiple-angles.safetensors"]},
#     "Photo-to-Anime": {
#         "repo": "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
#         "weights": "Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
#         "adapter_name": "photo-to-anime",
#         "prompt": "Transform into anime.",
#         "inputs": 1, "base": "2509"},
#     "Anime-V2": {
#         "repo": "prithivMLmods/Qwen-Image-Edit-2511-Anime",
#         "weights": "Qwen-Image-Edit-2511-Anime-2000.safetensors",
#         "adapter_name": "anime-v2",
#         "prompt": "Transform into anime (while preserving the background and remaining "
#                   "elements maintaining realism and original details.)",
#         "inputs": 1, "base": "2511"},
#     "Light-Migration": {
#         "repo": "dx8152/Qwen-Edit-2509-Light-Migration",
#         "weights": "参考色调.safetensors",
#         "adapter_name": "light-migration",
#         "prompt": "Refer to the color tone, remove the original lighting from Image 1, "
#                   "and relight Image 1 based on the lighting and color tone of Image 2.",
#         "inputs": 2, "base": "2509"},
#     "Upscaler": {
#         "repo": "starsfriday/Qwen-Image-Edit-2511-Upscale2K",
#         "weights": "qwen_image_edit_2511_upscale.safetensors",
#         "adapter_name": "upscale-2k",
#         "prompt": "Upscale this picture to 4K resolution.",
#         "inputs": 1, "base": "2511"},
#     "Style-Transfer": {
#         "repo": "zooeyy/Style-Transfer",
#         "weights": "Style Transfer-Alpha-V0.1.safetensors",
#         "adapter_name": "style-transfer",
#         "prompt": "Convert Image 1 to the style of Image 2.",
#         "inputs": 2, "base": "2511"},
#     "Manga-Tone": {
#         "repo": "nappa114514/Qwen-Image-Edit-2509-Manga-Tone",
#         "weights": "tone001.safetensors",
#         "adapter_name": "manga-tone",
#         "prompt": "Paint with manga tone.",
#         "inputs": 1, "base": "2509"},
#     "Anything2Real": {
#         "repo": "lrzjason/Anything2Real_2601",
#         "weights": "anything2real_2601.safetensors",
#         "adapter_name": "anything2real",
#         "prompt": "Change the picture to realistic photograph.",
#         "inputs": 1, "base": "2511"},
#     "Fal-Multiple-Angles": {
#         "repo": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
#         "weights": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
#         "adapter_name": "fal-multiple-angles",
#         "prompt": "Front-right quarter view.",
#         "inputs": 1, "base": "2511"},
#     "Polaroid-Photo": {
#         "repo": "prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo",
#         "weights": "Qwen-Image-Edit-2511-Polaroid-Photo.safetensors",
#         "adapter_name": "polaroid-photo",
#         "prompt": "cinematic polaroid with soft grain subtle vignette gentle lighting white "
#                   "frame handwritten photographed preserving realistic texture and details.",
#         "inputs": 1, "base": "2511"},
#     "Unblur-Anything": {
#         "repo": "prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale",
#         "weights": "Qwen-Image-Edit-Unblur-Upscale_15.safetensors",
#         "adapter_name": "unblur-anything",
#         "prompt": "Unblur and upscale.",
#         "inputs": 1, "base": "2511"},
#     "Midnight-Noir-Eyes-Spotlight": {
#         "repo": "prithivMLmods/Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight",
#         "weights": "Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight.safetensors",
#         "adapter_name": "midnight-noir-eyes-spotlight",
#         "prompt": "Transform into Midnight Noir Eyes Spotlight.",
#         "inputs": 1, "base": "2511"},
#     "Hyper-Realistic-Portrait": {
#         "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
#         "weights": "HRP_20.safetensors",
#         "adapter_name": "hyper-realistic-portrait",
#         "prompt": "Transform into a hyper-realistic face portrait.",
#         "inputs": 1, "base": "2511"},
#     "Ultra-Realistic-Portrait": {
#         "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
#         "weights": "URP_20.safetensors",
#         "adapter_name": "ultra-realistic-portrait",
#         "prompt": "Ultra-realistic portrait.",
#         "inputs": 1, "base": "2511"},
#     "Pixar-Inspired-3D": {
#         "repo": "prithivMLmods/Qwen-Image-Edit-2511-Pixar-Inspired-3D",
#         "weights": "PI3_20.safetensors",
#         "adapter_name": "pixar-inspired-3d",
#         "prompt": "Transform it into Pixar-inspired 3D.",
#         "inputs": 1, "base": "2511"},
#     "Noir-Comic-Book": {
#         "repo": "prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel",
#         "weights": "Noir-Comic-Book-Panel_20.safetensors",
#         "adapter_name": "noir-comic-book",
#         "prompt": "Transform into a noir comic book style.",
#         "inputs": 1, "base": "2511"},
#     "Any-Light": {
#         "repo": "lilylilith/QIE-2511-MP-AnyLight",
#         "weights": "QIE-2511-AnyLight_.safetensors",
#         "adapter_name": "any-light",
#         "prompt": "Apply the lighting from image 2 to image 1.",
#         "inputs": 2, "base": "2511"},
#     "Studio-DeLight": {
#         "repo": "prithivMLmods/QIE-2511-Studio-DeLight",
#         "weights": "QIE-2511-Studio-DeLight-5000.safetensors",
#         "adapter_name": "studio-delight",
#         "prompt": "Neutral uniform lighting. Preserve identity and composition.",
#         "inputs": 1, "base": "2511"},
#     "Cinematic-FlatLog": {
#         "repo": "prithivMLmods/QIE-2511-Cinematic-FlatLog-Control",
#         "weights": "QIE-2511-Cinematic-FlatLog-Control-3200.safetensors",
#         "adapter_name": "flat-log",
#         "prompt": "Transform into a cinematic flat log.",
#         "inputs": 1, "base": "2511"},
# }

SUBDIR = "_hf-edit"


def _apply_config_overrides(specs):
    """config 'edit_loras': dict nom -> spec (ajout/remplacement) ou null (retrait)."""
    extra = CONFIG.get("edit_loras")
    if not isinstance(extra, dict):
        return specs
    out = dict(specs)
    for name, spec in extra.items():
        if spec is None:
            out.pop(name, None)
        elif isinstance(spec, dict) and spec.get("repo") and spec.get("weights"):
            s = dict(spec)
            s.setdefault("adapter_name", name.lower().replace(" ", "-"))
            s.setdefault("prompt", "")
            s.setdefault("inputs", 1)
            out[name] = s
        else:
            _log(f"edit_loras['{name}'] ignored: needs 'repo' and 'weights'")
    return out


SPECS = _apply_config_overrides(EDIT_LORA_SPECS)


def names():
    """Noms des presets, dans l'ordre du registre."""
    return list(SPECS)


def spec(name):
    """Spec d'un preset (None si inconnu). Tolere l'adapter_name et la casse."""
    if not name:
        return None
    if name in SPECS:
        return SPECS[name]
    low = str(name).strip().lower()
    for n, s in SPECS.items():
        if n.lower() == low or s.get("adapter_name", "").lower() == low:
            return s
    return None


def canonical_name(name):
    """Nom du registre pour un nom/adapter_name (None si inconnu)."""
    s = spec(name)
    if s is None:
        return None
    for n, v in SPECS.items():
        if v is s:
            return n
    return None


def edit_loras_dir():
    """Dossier des LoRA d'edition telecharges: config 'edit_loras_dir', sinon
    <LORAS_DIR>/_hf-edit (LORAS_DIR lu a l'appel: l'UI peut le changer)."""
    d = (CONFIG.get("edit_loras_dir") or "").strip()
    if d:
        return d
    # LORAS_DIR SANS importer cz_pipeline (torch): caps du protocole doit rester
    # leger. Si le pipeline est deja charge (UI), sa valeur courante prime.
    import sys
    cp = sys.modules.get("cz_pipeline")
    base = getattr(cp, "LORAS_DIR", None) if cp is not None else None
    if not base:
        from cz_core import HERE, _prefs
        base = (os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
                or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))
    return os.path.join(base, SUBDIR)


def local_path(name):
    """Chemin local attendu du preset (existant ou non)."""
    s = spec(name)
    if s is None:
        return None
    return os.path.join(edit_loras_dir(), s["adapter_name"] + ".safetensors")


def _candidates(s):
    """Noms de fichier sous lesquels un preset peut deja exister dans une bibliotheque."""
    return ([s["adapter_name"] + ".safetensors", os.path.basename(s["weights"])]
            + list(s.get("local_names") or []))


def available_path(name, index=None):
    """Chemin local du preset s'il est deja sur disque (dossier _hf-edit OU une
    bibliotheque LoRA, ex. copie Civitai), sinon None. Ne telecharge jamais."""
    s = spec(name)
    if s is None:
        return None
    p = local_path(name)
    if os.path.isfile(p):
        return p
    return find_local(_candidates(s), index=index)


def is_downloaded(name, index=None):
    return available_path(name, index=index) is not None


def resolve(name, download=True, progress=None):
    """Chemin local du LoRA du preset; le telecharge du hub si absent (et
    download=True). Leve FileNotFoundError si absent et download=False,
    RuntimeError si le telechargement echoue."""
    s = spec(name)
    if s is None:
        raise KeyError(f"unknown edit LoRA preset: {name!r} (known: {', '.join(SPECS)})")
    dst = local_path(name)
    if os.path.isfile(dst):
        return dst
    # Deja dans une bibliotheque LoRA (principal ou extras, ex. copie Civitai) ?
    # -> on l'utilise sans rien telecharger.
    found = find_local(_candidates(s))
    if found:
        _dbg(f"edit LoRA {name}: using existing file {found}")
        return found
    if not download:
        raise FileNotFoundError(dst)
    return _download(s, dst, progress)


def lora_dirs():
    """Dossiers LoRA (principal + extras) SANS importer cz_pipeline si absent (torch):
    memes priorites env > preferences > config que cz_pipeline."""
    import sys
    cp = sys.modules.get("cz_pipeline")
    if cp is not None and hasattr(cp, "_lora_dirs"):
        return cp._lora_dirs()
    from cz_core import HERE, _prefs
    main = (os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
            or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))
    extra = (os.environ["LORAS_EXTRA_DIRS"] if "LORAS_EXTRA_DIRS" in os.environ
             else (_prefs.get("loras_extra_dirs") or CONFIG.get("loras_extra_dirs") or []))
    if isinstance(extra, str):
        extra = [p for chunk in extra.split(os.pathsep) for p in chunk.split(";")]
    dirs = [main]
    for d in extra:
        d = str(d or "").strip()
        if d and d not in dirs:
            dirs.append(d)
    return dirs


def local_index():
    """{nom_de_fichier_minuscule: chemin} de tous les .safetensors des dossiers LoRA
    (principal d'abord: il gagne sur un meme nom). Un seul parcours disque, a passer
    a find_local / is_downloaded / catalog quand on interroge plusieurs presets."""
    idx = {}
    for d in lora_dirs():
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                k = f.lower()
                if k.endswith(".safetensors") and k not in idx:
                    idx[k] = os.path.join(root, f)
    return idx


def find_local(filenames, index=None):
    """Premier fichier dont le nom (insensible a la casse) figure dans `filenames`,
    cherche dans les dossiers LoRA (index = local_index() deja construit). None si absent."""
    if index is None:
        index = local_index()
    for f in filenames:
        p = index.get(str(f).lower()) if f else None
        if p:
            return p
    return None


# ----------------------------------------------------------------------------
# Mode rapide: LoRA Lightning (distillation) pour l'edition, 4 ou 8 steps, CFG off.
# Le fichier depend de la revision du modele d'edition (2509 / 2511): choisi d'apres
# le nom du modele omni. Cherche d'abord dans les dossiers LoRA (les bibliotheques
# Civitai ont souvent deja la 8-steps), sinon telecharge (repo lightx2v, 2509 = gated:
# un hf_token est necessaire).
# ----------------------------------------------------------------------------
AUTO_SPEED = "Auto (model profile)"
# Presets Lightning: VIDES chez klein. Ces LoRA sont des accelerateurs pour
# Qwen-Image-Edit (architecture differente) ET klein est DEJA distille a 4 steps:
# il n'y a rien a accelerer. 'Edit speed' se reduit donc a 'Off'.
SPEED_SPECS = {}

# --- Reference de merge (amont Qwen). Ne PAS reactiver: incompatible FLUX.2.
# SPEED_SPECS = {
#     "Lightning 4 steps": {
#         "steps": 4, "guidance": 1.0,
#         "files": {
#             "2509": ("lightx2v/Qwen-Image-Edit-2509-Lightning",
#                      "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"),
#             "2511": ("lightx2v/Qwen-Image-Edit-2511-Lightning",
#                      "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")}},
#     "Lightning 8 steps": {
#         "steps": 8, "guidance": 1.0,
#         "files": {
#             "2509": ("lightx2v/Qwen-Image-Edit-2509-Lightning",
#                      "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"),
#             "2511": ("lightx2v/Qwen-Image-Edit-2511-Lightning",
#                      "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")}},
# }


def speed_names():
    """Choix du dropdown 'Edit speed' (sans 'Off', ajoute par cz_pipeline).

    Vide chez klein: le modele est deja distille a 4 steps et il n'existe pas de LoRA
    Lightning FLUX.2 a empiler. Le dropdown se reduit donc a 'Off' cote cz_pipeline,
    et caps.edit_fast n'annonce plus de mode qui echouerait."""
    return []


def edit_base_revision(omni_model):
    """'2511' si le modele d'edition est un 2511, sinon '2509' (Plus)."""
    return "2511" if "2511" in str(omni_model or "") else "2509"


def resolve_speed(name, omni_model, download=True):
    """{"name", "steps", "guidance", "path", "base"} pour un mode Lightning: la LoRA
    est prise dans les dossiers LoRA si presente, sinon telechargee dans edit_loras_dir.
    KeyError si le nom est inconnu."""
    key = None
    for k in SPEED_SPECS:
        if k.lower() == str(name or "").strip().lower():
            key = k
    if key is None:
        raise KeyError(f"unknown edit speed: {name!r} (known: {', '.join(SPEED_SPECS)})")
    sp = SPEED_SPECS[key]
    base = edit_base_revision(omni_model)
    repo, fname = sp["files"][base]
    path = find_local([fname])
    if path is None:
        dst = os.path.join(edit_loras_dir(), fname)
        if os.path.isfile(dst):
            path = dst
        elif download:
            path = _download({"repo": repo, "weights": fname}, dst)
        else:
            raise FileNotFoundError(dst)
    return {"name": key, "steps": int(sp["steps"]), "guidance": float(sp["guidance"]),
            "path": path, "base": base}


def _download(s, dst, progress=None):
    """hf_hub_download -> copie atomique vers dst (nom ASCII stable)."""
    import shutil
    from huggingface_hub import hf_hub_download
    from cz_core import _apply_hf_token
    _apply_hf_token(CONFIG.get("hf_token"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    _log(f"edit LoRA: downloading {s['repo']}/{s['weights']} -> {dst}")
    if progress:
        progress(0.0, f"Downloading {s['repo']} ...")
    try:
        src = hf_hub_download(s["repo"], s["weights"])
    except Exception as e:
        raise RuntimeError(f"download failed for {s['repo']}/{s['weights']}: {e}") from e
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    _dbg(f"edit LoRA stored: {dst} ({os.path.getsize(dst) / 1024**2:.0f} MB)")
    if progress:
        progress(1.0, "Downloaded.")
    return dst


def status_label(name, index=None):
    """Libelle pour l'UI: 'Nom ✓' si deja sur disque (dossier _hf-edit ou une
    bibliotheque LoRA), 'Nom ⬇' sinon."""
    return f"{name} {'✓' if is_downloaded(name, index=index) else '⬇'}"


def strip_label(label):
    """Inverse de status_label (le dropdown renvoie le libelle)."""
    if not label:
        return ""
    return str(label).rstrip(" ✓⬇").strip()


def catalog():
    """Liste legere pour caps / UI: [{name, adapter_name, repo, prompt, inputs,
    base, downloaded}] - sans rien telecharger."""
    idx = local_index()
    return [{"name": n, "adapter_name": s["adapter_name"], "repo": s["repo"],
             "prompt": s.get("prompt", ""), "inputs": int(s.get("inputs", 1)),
             "base": s.get("base", ""), "downloaded": is_downloaded(n, index=idx)}
            for n, s in SPECS.items()]
