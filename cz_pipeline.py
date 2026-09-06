"""crispz-klein - coeur FLUX.2 Klein (diffusers, BF16): chargement des pipelines
(txt2img / img2img / inpaint) + edition multi-reference (onglet Omni/Edit) + LoRA /
checkpoints / transformer, generation et orchestration (generate / txt2img_run /
process_one / outpaint / inpaint) + l'etat mutable runtime.

Fork de crispz-qwen-edit (Qwen-Image). Mapping :
  - base txt2img             -> Flux2KleinPipeline
  - onglet Omni/Edit         -> Flux2KleinPipeline  (MEME objet: `image` accepte une
                                LISTE de PIL -> multi-reference natif, pas de 2e modele)
  - inpaint / reframe        -> Flux2KleinInpaintPipeline
  - img2img (refine/upscale) -> Flux2KleinInpaintPipeline + masque BLANC plein
                                (le pipeline base n'expose PAS `strength`)

klein-4B est DISTILLE (`is_distilled: true`). Mesure du 2026-09-05 (tests/
test_klein_guidance.py, RTX 5090): guidance_scale 1.0 / 4.0 / 8.0 -> images
bit-a-bit identiques (MAE 0.0000), diffusers emettant lui-meme "Guidance scale
is ignored for step-wise distilled models". Il n'y a donc NI CFG NI negative
prompt utilisables: `_cfg` renvoie {} et le protocole annonce
supports.negative = False. Le curseur "guidance" de l'UI est conserve (contrat
d'API cz_ui) mais n'a aucun effet sur le rendu.

L'API publique du module reste identique a l'amont (memes noms, ex.
ZIMAGE_TRANSFORMER, generate_omni, OMNI_MODEL, SAMPLER_CHOICES) pour ne casser ni
cz_ui ni cz_cli ni cz_protocol. Les symboles "omni" survivent mais pointent
desormais sur le MEME modele que le base.

app lit l'etat courant via cz_pipeline.NAME (BASE_REPO, ZIMAGE_TRANSFORMER, ...) et pose
cz_pipeline._PROGRESS / cz_pipeline._STOP depuis les handlers UI.
Ne depend que de cz_core / cz_esrgan / cz_imageio (jamais de app ni de gradio).
"""

import os
import gc
import sys
import time
import json
import threading

import numpy as np
import torch
from PIL import Image

import cz_core
from cz_core import (
    CONFIG, HERE, DEVICE, DTYPE,
    DEFAULT_TILE, DEFAULT_OVERLAP, DEFAULT_REFINE_TILE, DEFAULT_REFINE_OVERLAP,
    _prefs, _is_single_file, _log, _dbg,
)

# Modele FLUX.2 Klein de base (txt2img/img2img/inpaint/edit). Surcharge via env
# ZIMAGE_MODEL (compat) ou KLEIN_MODEL, ou prefs. Repo public, Apache 2.0.
# ATTENTION: ne PAS basculer sur FLUX.2-klein-9B (licence non commerciale), cf. FORK.md.
DEFAULT_BASE_REPO = (os.environ.get("KLEIN_MODEL") or "black-forest-labs/FLUX.2-klein-4B")
# L'edition multi-reference n'a PAS de modele separe chez klein: `image` du pipeline de
# base accepte une liste de PIL. Le defaut "omni" est donc le modele de base lui-meme
# (le symbole survit pour cz_ui / cz_protocol, cf. docstring).
DEFAULT_OMNI_REPO = DEFAULT_BASE_REPO
from cz_esrgan import load_esrgan, esrgan_upscale
from cz_imageio import _now_stamp

# Vitesse: autorise TF32 (matmul/cudnn) sur GPU. Gain gratuit sur Ampere+ pour les
# operations fp32 residuelles; les poids restent BF16. Sans effet hors CUDA.
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# Modele Z-Image courant. Un repo HF / dossier diffusers -> BASE_REPO. Un fichier
# single-file (.safetensors Civitai) passe comme "modele" -> transformer override
# (le VAE et l'encodeur Qwen3 restent tires du repo de base).
_zmodel = os.environ.get("ZIMAGE_MODEL") or _prefs.get("zimage_model") or DEFAULT_BASE_REPO
ZIMAGE_TRANSFORMER = os.environ.get("ZIMAGE_TRANSFORMER") or _prefs.get("zimage_transformer") or None
if _is_single_file(_zmodel):
    ZIMAGE_TRANSFORMER = _zmodel
    BASE_REPO = DEFAULT_BASE_REPO
else:
    BASE_REPO = _zmodel

# Dossiers de modeles Z-Image: checkpoints single-file a switcher + LoRA a appliquer.
CHECKPOINTS_DIR = (os.environ.get("CHECKPOINTS_DIR") or _prefs.get("checkpoints_dir")
                   or CONFIG.get("checkpoints_dir") or os.path.join(HERE, "checkpoints"))
# Dossier checkpoints supplementaire (optionnel) -> fusionne avec CHECKPOINTS_DIR dans
# la meme liste de checkpoints. Vide par defaut; configurable via UI / prefs / config / env.
CHECKPOINTS_EXTRA_DIR = (os.environ.get("CHECKPOINTS_EXTRA_DIR") or _prefs.get("checkpoints_extra_dir")
                         or CONFIG.get("checkpoints_extra_dir") or "").strip()
LORAS_DIR = (os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
             or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))


def _split_dirs(spec):
    """Liste de dossiers depuis une liste JSON ou une chaine 'a;b' (os.pathsep ou ';')."""
    if not spec:
        return []
    if isinstance(spec, str):
        parts = [p for chunk in spec.split(os.pathsep) for p in chunk.split(";")]
    else:
        parts = list(spec)
    out = []
    for p in parts:
        p = str(p or "").strip()
        if p and p not in out:
            out.append(p)
    return out


# Dossiers LoRA SUPPLEMENTAIRES (ex. la bibliotheque Civitai partagee avec d'autres
# outils): env LORAS_EXTRA_DIRS ('a;b') > preferences > config 'loras_extra_dirs'.
# Fusionnes avec LORAS_DIR dans une seule liste; en cas de meme nom, LORAS_DIR gagne.
LORAS_EXTRA_DIRS = _split_dirs(os.environ["LORAS_EXTRA_DIRS"] if "LORAS_EXTRA_DIRS" in os.environ
                               else (_prefs.get("loras_extra_dirs")
                                     or CONFIG.get("loras_extra_dirs")))


def _lora_dirs():
    """Dossiers LoRA a scanner: principal + extras, sans doublon, dans l'ordre de priorite."""
    dirs = [LORAS_DIR]
    for d in LORAS_EXTRA_DIRS:
        if d and d not in dirs:
            dirs.append(d)
    return dirs


def resolve_lora_path(name):
    """Chemin d'une LoRA depuis un nom de slot: chemin absolu tel quel, sinon nom relatif
    (avec sous-dossiers) cherche dans LORAS_DIR puis les extras. Si absent partout, le
    chemin dans LORAS_DIR (le caller signale 'not found')."""
    name = str(name or "")
    if os.path.isabs(name):
        return name
    for d in _lora_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(LORAS_DIR, name)


# LoRA actives: liste de (chemin, poids). Plusieurs LoRA combinables (multi-slots).
LORAS = []
LORA_WEIGHT = float(CONFIG.get("default_lora_weight", 1.0))  # poids par defaut des slots


def _lora_weight_range():
    """Bornes des curseurs de poids LoRA (config 'lora_weight_min'/'lora_weight_max').
    Defaut -2..2: les poids NEGATIFS sont valides et utiles (ils inversent l'effet de la
    LoRA). Defensif: valeurs illisibles ou min >= max -> on retombe sur le defaut."""
    try:
        lo = float(CONFIG.get("lora_weight_min", -2.0))
        hi = float(CONFIG.get("lora_weight_max", 2.0))
    except (TypeError, ValueError):
        _log("lora_weight_min/max: not a number, using -2..2")
        return -2.0, 2.0
    if lo >= hi:
        _log(f"lora_weight_min ({lo}) >= lora_weight_max ({hi}), using -2..2")
        return -2.0, 2.0
    return lo, hi


LORA_WEIGHT_MIN, LORA_WEIGHT_MAX = _lora_weight_range()
# Le poids par defaut doit rester dans les bornes (sinon le curseur naitrait hors plage).
LORA_WEIGHT = min(LORA_WEIGHT_MAX, max(LORA_WEIGHT_MIN, LORA_WEIGHT))
# LoRA appliquees AU DEMARRAGE (ex. Lightning 8-step). config 'default_loras' = liste de
# noms (dans LORAS_DIR) ou de paires [nom, poids]. Resolues en (chemin, poids).
for _spec in (CONFIG.get("default_loras") or []):
    _nm, _w = (_spec if isinstance(_spec, (list, tuple)) and len(_spec) == 2
               else (_spec, LORA_WEIGHT))
    if _nm and _nm not in ("None", "none"):
        _p = resolve_lora_path(_nm)
        if os.path.isfile(_p):
            LORAS.append((_p, float(_w)))
# Modele Omni/Edit. Chez klein il n'y a PAS de second modele: l'edition multi-reference
# est servie par le pipeline de base. OMNI_MODEL suit donc BASE_REPO et existe surtout
# pour que cz_ui / cz_protocol gardent leur contrat (toujours non vide -> edit dispo).
OMNI_MODEL = BASE_REPO

# Caches process-wide. Un pipeline "base" (txt2img ZImagePipeline) detient les
# composants; img2img / inpaint en derivent via from_pipe -> poids partages, pas de
# VRAM en double. Clef de cache = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE, LORAS).
_BASE_PIPE = None
_DERIVED = {}
_LOADED_KEY = None
# LoRA reellement posees sur _BASE_PIPE (liste de (chemin, poids)). Sert a echanger les
# LoRA a chaud sans recharger le modele: si ca diverge de LORAS, _apply_loras resynchronise.
_APPLIED_LORAS = []
# LoRA d'EDITION (pipe omni / Qwen-Image-Edit): jeu SEPARE du base, car le transformer
# d'edition est un autre modele (les presets cz_edit_loras visent 2509/2511). Meme
# format (chemin, poids). EDIT_LORAS_ENABLED = la case "Edit LoRAs" de l'UI: OFF -> le
# jeu est memorise mais pas pose (permet de comparer avec/sans en un clic).
EDIT_LORAS = []
EDIT_LORAS_ENABLED = bool(CONFIG.get("edit_loras_enabled", True))
_APPLIED_EDIT_LORAS = []
# Mode RAPIDE de l'edition (dropdown 'Edit speed'): None = off (steps/guidance des
# Settings), sinon {"name", "steps", "guidance", "path"} - path = LoRA Lightning a
# empiler sur le pipe d'edition (None pour 'Auto': un modele deja distille, Rapid-AIO
# ou merge Lightning, dont le profil model_profiles fixe steps/guidance).
EDIT_SPEED = None

# Palier 2 (cohabitation VRAM): offload CPU de la passe diffusion. none = tout en VRAM.
# model = decharge par sous-module (bon compromis). sequential = plus agressif, plus lent.
# N'est PAS de la quantif: les poids restent BF16, ils transitent RAM <-> GPU.
# Qwen-Image est gros (~20B) : on initialise depuis la config (default_cpu_offload, defaut
# 'model' pour ce fork) ou l'env CZ_OFFLOAD -> offload actif DES le 1er chargement (anti-OOM).
OFFLOAD_CHOICES = ("none", "model", "sequential")
OFFLOAD_MODE = (os.environ.get("CZ_OFFLOAD") or CONFIG.get("default_cpu_offload") or "none")
if OFFLOAD_MODE not in OFFLOAD_CHOICES:
    OFFLOAD_MODE = "none"

# Guidance. klein-4B est DISTILLE: diffusers IGNORE guidance_scale (verifie, cf.
# docstring + tests/test_klein_guidance.py -> images bit-a-bit identiques de 1.0 a 8.0).
# La variable est conservee parce que cz_ui / cz_cli / cz_protocol la lisent et
# l'affichent, mais _cfg() ne la transmet PLUS au pipeline: elle n'a aucun effet.
# Override possible via env KLEIN_CFG (sans effet non plus, garde pour la symetrie).
GUIDANCE = float(os.environ.get("KLEIN_CFG") or CONFIG.get("default_guidance") or 0) or 1.0

# Force ratio (facon Fooocus) pour upscale/img2img: si defini, l'image d'ENTREE est
# recadree au centre a ce ratio avant traitement (crop to fit). Vide = ratio natif preserve
# (defaut). Format: 'W:H' ou 'WxH' (ex. '13:19', '832x1216'). Pilotable par l'UI (case a
# cocher + dropdown Aspect ratio) via set_force_ratio, ou par config.txt 'force_upscale_ratio'.
FORCE_RATIO = (os.environ.get("CZ_FORCE_RATIO") or CONFIG.get("force_upscale_ratio") or "").strip()
# Comment atteindre le ratio force: 'crop' = recadrage centre (perd les bords, defaut),
# 'extend' = etend l'image au ratio par outpaint (ne perd rien, ajoute des bandes
# generees). UI (radio) via set_force_ratio_mode, config 'force_ratio_mode'.
FORCE_RATIO_MODE = (os.environ.get("CZ_FORCE_RATIO_MODE")
                    or CONFIG.get("force_ratio_mode") or "crop").strip().lower()
# Passe de fusion des raccords du mode extend: apres l'outpaint des bandes, une passe
# img2img LEGERE tourne sur l'image etendue et SEULES les bandes + une marge de
# transition feather sont recollees depuis elle (centre original intact). 0 = desactive.
try:
    EXTEND_DENOISE = float(CONFIG.get("force_ratio_extend_denoise", 0.22) or 0.0)
except Exception:
    EXTEND_DENOISE = 0.22

# Sampler / scheduler. Le pipeline Z-Image impose un schedule `sigmas` custom: seuls
# les schedulers dont set_timesteps accepte `sigmas` fonctionnent. En pratique -> Euler
# flow-matching (natif, defaut), UniPC (multistep) et LCM flow-matching (interessant sur
# les modeles distilles/Turbo: peu de steps, guidance ~0-1).
# Les DPM++ 2M / DPM2a / DPM++ SDE (dpmpp_sde) de diffusers ne prennent PAS de sigmas
# custom -> incompatibles (DPMSolverSDEScheduler exige en plus torchsde). Non exposes.
SAMPLER_CHOICES = ("euler", "unipc", "lcm")
SAMPLER = (os.environ.get("ZIMAGE_SAMPLER") or CONFIG.get("default_sampler") or "euler").strip().lower()
if SAMPLER not in SAMPLER_CHOICES:
    SAMPLER = "euler"

# Schedule de sigmas (= le "scheduler" facon ComfyUI). sgm_uniform = natif Z-Image
# (linspace + dynamic shift). beta/karras/exponential = re-mapping des sigmas applique
# PAR-DESSUS le schedule du pipeline (FlowMatchEuler/UniPC: use_*_sigmas). beta -> scipy.
SCHEDULE_CHOICES = ("sgm_uniform", "beta", "karras", "exponential")
# 'simple' (ComfyUI) designe EXACTEMENT le schedule natif expose ici sous 'sgm_uniform':
# les sigmas par defaut que le pipeline passe au scheduler sont linspace(1, 1/n, n),
# ce que ComfyUI appelle 'simple' sur un modele flow-matching. Accepte
# en entree partout (config/env/CLI/XYZ) pour recopier une recette CivitAI au mot pres,
# mais normalise vers le nom canonique: metadonnees et presets ne portent qu'un seul nom.
_SCHEDULE_ALIASES = {"simple": "sgm_uniform"}
SCHEDULE_INPUTS = SCHEDULE_CHOICES + tuple(_SCHEDULE_ALIASES)   # listes ouvertes (CLI/XYZ)


def _norm_schedule(name, default="sgm_uniform"):
    """Nom de schedule -> nom canonique (alias resolus). Inconnu -> `default`."""
    n = (name or "").strip().lower()
    n = _SCHEDULE_ALIASES.get(n, n)
    return n if n in SCHEDULE_CHOICES else default


SCHEDULE = _norm_schedule(os.environ.get("ZIMAGE_SCHEDULE") or CONFIG.get("default_schedule"))
_SCHEDULE_FLAG = {"beta": "use_beta_sigmas", "karras": "use_karras_sigmas",
                  "exponential": "use_exponential_sigmas"}  # sgm_uniform -> aucun flag (natif)
# Config natif du scheduler du modele (capture au 1er chargement) -> base de construction
# des autres samplers (conserve shift/flow params quel que soit le sampler courant).
_BASE_SCHED_CONFIG = None

# Hook de progression UI (gradio gr.Progress). None hors UI (CLI/serveur). Pose par
# les handlers via cz_pipeline._PROGRESS = ...
_PROGRESS = None
# Stop "facon Fooocus": flag global + interruption des pipelines diffusers. Pose par
# les handlers via cz_pipeline._STOP = ... et par request_stop().
_STOP = False

# Verrou GPU: serialise TOUTES les generations. Gradio ne serialise pas les events de
# LISTENERS differents (Generate manuel vs Run queue vs detaileur): deux threads peuvent
# alors appeler le MEME pipeline partage et stepper le MEME scheduler -> son index
# depasse la fin ("IndexError: index 31 is out of bounds for dimension 0 with size 31",
# scheduling_flow_match_euler_discrete.step). RLock: les imbrications d'un meme thread
# (txt2img_run -> generate, process_one -> _refine_whole) restent libres.
_GPU_LOCK = threading.RLock()


def _gpu_serial(fn):
    """Decorateur: execute fn sous _GPU_LOCK (une seule generation GPU a la fois)."""
    import functools

    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        with _GPU_LOCK:
            return fn(*args, **kwargs)
    return _locked

# Gestion du seed (facon Fooocus):
#  _LAST_SEED         = seed CONCRET du dernier rendu (un -1 aleatoire est resolu en
#                       valeur reelle) -> bouton "Reuse last seed" + metadonnees justes.
#  _NO_SEED_INCREMENT = True -> tout un batch utilise le meme seed (pas de +i par image).
_LAST_SEED = -1
_NO_SEED_INCREMENT = False
# True -> en txt2img+upscale, sauve AUSSI l'image txt2img d'origine (avant l'upscale).
_SAVE_PRE_UPSCALE = bool(CONFIG.get("save_pre_upscale", False))


def set_no_seed_increment(v):
    global _NO_SEED_INCREMENT
    _NO_SEED_INCREMENT = bool(v)


def set_save_pre_upscale(v):
    global _SAVE_PRE_UPSCALE
    _SAVE_PRE_UPSCALE = bool(v)



def set_guidance(g):
    global GUIDANCE
    GUIDANCE = float(g)


def _cfg(negative=None, guidance=None):
    """kwargs CFG. Chez klein: AUCUN.

    klein-4B est step-wise distilled -> diffusers ignore `guidance_scale` (mesure:
    1.0 / 4.0 / 8.0 donnent des images bit-a-bit identiques, cf.
    tests/test_klein_guidance.py) et les pipelines Flux2Klein* n'exposent PAS
    `negative_prompt` (seulement `negative_prompt_embeds`, inutilisable sans CFG).

    On renvoie donc {} plutot que de transmettre des kwargs sans effet ou de faire
    avertir diffusers a chaque appel. La signature est conservee: tous les callsites
    de l'amont (`**_cfg(negative)`) restent valides, et le negative eventuellement
    fourni est ignore SILENCIEUSEMENT ici mais ANNONCE en amont par
    cz_protocol (supports.negative = False + warning sur un spec qui en porte un)
    -- regle maison: degradation annoncee, jamais silencieuse."""
    if negative:
        _dbg("negative prompt ignore: klein est distille (ni CFG ni negative_prompt)")
    return {}


def _qwen_call(pipe, **kw):
    """Appelle un pipeline Flux2Klein en absorbant les deux ecarts d'API avec l'amont.

    1. Masque blanc implicite: `Flux2KleinPipeline` n'expose PAS `strength`, donc
       l'img2img passe par `Flux2KleinInpaintPipeline`. Un appel qui porte `image` +
       `strength` SANS `mask_image` est un img2img -> on injecte un masque
       entierement BLANC (tout est redessine) de la taille de l'image.
    2. Kwargs CFG residuels: si un callsite (ou un merge amont) repasse
       `true_cfg_scale` / `negative_prompt`, on les retire et on relance au lieu de
       crasher la generation.

    Le nom `_qwen_call` est conserve pour limiter la surface de conflit au merge
    depuis `qwen/main` (une trentaine de callsites)."""
    # guidance_scale: klein est distille -> diffusers IGNORE toute valeur > 1.0 et
    # log un warning a CHAQUE appel. On passe donc explicitement 1.0 (= pas de CFG,
    # la verite du modele) pour taire le bruit. Si un jour un checkpoint FLUX.2 NON
    # distille est charge, is_distilled est faux et on transmet le curseur de l'UI,
    # qui redevient un vrai CFG.
    if "guidance_scale" not in kw:
        try:
            distilled = bool(getattr(pipe.config, "is_distilled", False))
        except Exception:
            distilled = True
        kw["guidance_scale"] = 1.0 if distilled else float(GUIDANCE)
    if "strength" in kw and kw.get("image") is not None and "mask_image" not in kw:
        img = kw["image"]
        ref = img[0] if isinstance(img, (list, tuple)) else img
        kw["mask_image"] = Image.new("L", ref.size, 255)
        _dbg(f"img2img -> inpaint pipeline + masque blanc plein {ref.size}")
    try:
        return pipe(**kw)
    except TypeError as e:
        if any(k in kw for k in ("true_cfg_scale", "negative_prompt")):
            for k in ("true_cfg_scale", "negative_prompt"):
                kw.pop(k, None)
            _dbg(f"klein call: retry sans kwargs CFG ({e})")
            return pipe(**kw)
        raise


def _scheduler_accepts_sigmas(sched):
    """Le pipeline Z-Image appelle set_timesteps(..., sigmas=<schedule custom>). Un
    scheduler dont set_timesteps n'accepte pas `sigmas` plante a la generation."""
    import inspect
    try:
        return "sigmas" in inspect.signature(sched.set_timesteps).parameters
    except Exception:
        return False


def _build_scheduler(sampler, schedule, config):
    """Construit le scheduler choisi (sampler x schedule) depuis le config natif du modele.
    schedule (sgm_uniform/beta/karras/exponential) = remapping des sigmas (use_*_sigmas)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    kw = {}
    flag = _SCHEDULE_FLAG.get((schedule or "").lower())
    if flag:
        kw[flag] = True
    name = (sampler or "euler").lower()
    if name == "unipc":
        from diffusers import UniPCMultistepScheduler
        try:
            return UniPCMultistepScheduler.from_config(config, use_flow_sigmas=True, **kw)
        except Exception:
            return UniPCMultistepScheduler.from_config(config, **kw)
    if name == "lcm":
        # LCM flow-matching: accepte les sigmas custom du pipeline ET les flags de
        # schedule. Repli sur Euler si la version de diffusers ne l'expose pas.
        try:
            from diffusers import FlowMatchLCMScheduler
            return FlowMatchLCMScheduler.from_config(config, **kw)
        except Exception as e:
            _log(f"sampler 'lcm' unavailable ({e}); falling back to euler")
    return FlowMatchEulerDiscreteScheduler.from_config(config, **kw)


def _apply_sampler(pipe):
    """Pose le scheduler courant (SAMPLER x SCHEDULE) sur un pipe. Verifie la compatibilite
    (sigmas custom) et retombe sur Euler/sgm_uniform si KO -> jamais de crash a la generation."""
    if _BASE_SCHED_CONFIG is None:
        return
    from diffusers import FlowMatchEulerDiscreteScheduler
    try:
        sched = _build_scheduler(SAMPLER, SCHEDULE, _BASE_SCHED_CONFIG)
        if not _scheduler_accepts_sigmas(sched):
            raise ValueError(f"{type(sched).__name__} n'accepte pas les sigmas custom de Z-Image")
        pipe.scheduler = sched
        _dbg(f"sampler applied: {SAMPLER}/{SCHEDULE} -> {type(pipe.scheduler).__name__}")
    except Exception as e:
        _log(f"sampler '{SAMPLER}/{SCHEDULE}' incompatible ({e}); fallback Euler/sgm_uniform")
        try:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(_BASE_SCHED_CONFIG)
        except Exception:
            pass


def _reapply_sampler_all():
    """Re-applique le scheduler courant a tous les pipes en cache (base + derives)."""
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            _apply_sampler(p)


def set_sampler(name):
    """Change le sampler (euler/unipc) et le re-applique aux pipes en cache (pas de
    rechargement). Pas d'effet sur le pipe Omni (scheduler propre)."""
    global SAMPLER
    name = (name or "euler").strip().lower()
    if name not in SAMPLER_CHOICES:
        name = "euler"
    if name != SAMPLER:
        SAMPLER = name
        _reapply_sampler_all()
        _log(f"sampler -> {SAMPLER}")
    return f"Sampler: {SAMPLER} / {SCHEDULE}"


def set_schedule(name):
    """Change le schedule de sigmas (sgm_uniform/beta/karras/exponential, alias 'simple'
    = sgm_uniform) et le re-applique aux pipes en cache."""
    global SCHEDULE
    name = _norm_schedule(name)
    if name != SCHEDULE:
        SCHEDULE = name
        _reapply_sampler_all()
        _log(f"schedule -> {SCHEDULE}")
    return f"Sampler: {SAMPLER} / {SCHEDULE}"


def _progress(frac, desc=""):
    if _PROGRESS is not None:
        try:
            _PROGRESS(min(1.0, max(0.0, float(frac))), desc)
        except Exception:
            pass


# ---- Feedback de chargement des modeles (terminal + UI) ----
# from_pretrained est bloquant et silencieux (le 1er chargement telecharge depuis HF ->
# plusieurs minutes). On execute le chargement dans un thread et on rafraichit toutes les
# ~2s une ligne terminal + la barre Gradio (temps ecoule + VRAM allouee). Config bloc
# "load_progress"; enabled=false -> chargement direct (aucun thread, zero cout).
_LOAD_CFG = CONFIG.get("load_progress") if isinstance(CONFIG.get("load_progress"), dict) else {}
LOAD_PROGRESS_ENABLED = bool(_LOAD_CFG.get("enabled", True))
_LOAD_TARGET_GB = float(_LOAD_CFG.get("target_vram_gb", 14.0))
_LOAD_HEARTBEAT = float(_LOAD_CFG.get("heartbeat_s", 2.0))


def _fmt_load(label, elapsed, vram_gb):
    """Texte de progression de chargement (pur, testable). VRAM > 0 -> phase chargement
    en memoire; sinon phase download/lecture disque."""
    if vram_gb > 0.05:
        return f"{label}... {elapsed:.0f}s | {vram_gb:.1f} GB in VRAM"
    return f"{label}... {elapsed:.0f}s (downloading / reading, first run only)"


def _load_pct(elapsed, vram_gb, target_gb=None):
    """% honnete: base sur la VRAM allouee / cible une fois le chargement en memoire
    commence (plafonne 0.95); pendant le download (VRAM~0) petite barre temporelle."""
    target_gb = target_gb or _LOAD_TARGET_GB
    if vram_gb <= 0.05:
        return min(0.12, elapsed / 600.0)
    return min(0.95, vram_gb / max(1.0, float(target_gb)))


def _load_monitor(label, fn):
    """Execute fn() (chargement bloquant) dans un thread et rafraichit terminal + UI
    (temps + VRAM) toutes les ~2s. Renvoie le resultat de fn (releve son exception)."""
    if not LOAD_PROGRESS_ENABLED:
        return fn()
    box = {}

    def _work():
        try:
            box["v"] = fn()
        except BaseException as e:   # noqa: BLE001 - on re-leve dans le thread principal
            box["e"] = e

    th = threading.Thread(target=_work, daemon=True)
    t0 = time.time()
    th.start()
    while True:
        th.join(timeout=_LOAD_HEARTBEAT)
        el = time.time() - t0
        vram = (torch.cuda.memory_allocated() / 1024 ** 3) if DEVICE == "cuda" else 0.0
        line = _fmt_load(label, el, vram)
        if cz_core.LOG_LEVEL >= 1:
            sys.stderr.write("\r[crispz][load] " + line + "        ")
            sys.stderr.flush()
        _progress(_load_pct(el, vram), "Loading " + line)
        if not th.is_alive():
            break
    if cz_core.LOG_LEVEL >= 1:
        sys.stderr.write("\n")
        sys.stderr.flush()
    if "e" in box:
        raise box["e"]
    return box.get("v")


def request_stop():
    """Demande l'arret: stoppe la boucle de debruitage en cours (pipe._interrupt) et
    les boucles batch/tuiles (_STOP). Quasi-immediat (s'arrete au pas suivant)."""
    global _STOP
    _STOP = True
    n = 0
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            try:
                p._interrupt = True
                n += 1
            except Exception:
                pass
    _log(f"STOP requested (interrupt set on {n} pipeline(s))")
    return "Stopping..."


def set_zimage_model(repo_or_path):
    """Change le modele Klein. Un repo HF / dossier diffusers -> BASE_REPO.
    Un fichier single-file (.safetensors Civitai, .gguf) -> transformer override."""
    global BASE_REPO, ZIMAGE_TRANSFORMER
    if not repo_or_path:
        return
    if _is_single_file(repo_or_path):
        # Changement de transformer seul: PAS de free_vram -> _ensure_base echangera
        # uniquement le transformer (VAE + encodeur texte gardes en VRAM).
        if repo_or_path != ZIMAGE_TRANSFORMER:
            ZIMAGE_TRANSFORMER = repo_or_path
            _log("Klein transformer (single-file) changed -> transformer swap on next run")
    elif repo_or_path != BASE_REPO:
        # Le repo de base change: VAE/encodeur/tokenizer changent aussi -> reload complet.
        BASE_REPO = repo_or_path
        free_vram()
        _log("Klein base repo changed -> will reload")


def set_zimage_transformer(path):
    """Definit (ou enleve avec '' / None) le transformer single-file.

    NE libere PAS le pipeline: a repo de base identique, _ensure_base ne rechargera que
    le transformer (_swap_transformer) et gardera VAE + encodeur texte en VRAM."""
    global ZIMAGE_TRANSFORMER
    path = path or None
    if path != ZIMAGE_TRANSFORMER:
        ZIMAGE_TRANSFORMER = path
        _log(f"Klein transformer -> {path or '(repo de base)'} "
             "-> transformer swap on next run (base components kept)")


def _safetensors_header(path):
    """En-tete JSON d'un .safetensors (noms/dtypes/shapes des tenseurs, JAMAIS les
    poids) -- lecture de quelques centaines de Ko au plus, meme sur un fichier de 12 Go."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(min(n, 10_000_000)).decode("utf-8", "ignore"))


def _safetensors_unsupported(path):
    """Renvoie une raison (str) si le .safetensors n'est PAS chargeable, sinon None.
    Lit juste l'en-tete (rapide). Deux cas restent non supportes:
      - fichier LoRA range dans le dossier checkpoints (cles kohya/peft)
      - SVDQuant / Nunchaku (tenseurs nommes '*.qweight'): poids pre-quantifies INT4
        qui exigent le runtime nunchaku (kernels dedies), pas dequantifiables ici.
    Les FP8 / INT8 'scaled' facon ComfyUI ne sont PLUS rejetes: ils passent par le
    loader dequant (_safetensors_dequant + _load_dequant_state_dict)."""
    try:
        hdr = _safetensors_header(path)
        has_qweight = False
        lora_keys = 0
        te_keys = 0
        dit_keys = 0
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            if k.endswith(".qweight"):
                has_qweight = True
            # Encodeur texte Qwen2.5-VL (fichier ComfyUI 'qwen_2.5_vl_7b_fp8_scaled'):
            # couches LLM + tour visuelle, jamais de blocs de diffusion.
            if k.startswith(("model.layers.", "visual.", "lm_head.", "model.embed_tokens",
                             "language_model.", "model.language_model.")):
                te_keys += 1
            if "transformer_blocks" in k or k.startswith(("img_in", "txt_in")):
                dit_keys += 1
            if (".lora_down." in k or ".lora_up." in k or ".lora_A." in k
                    or ".lora_B." in k or k.startswith(("lora_unet_", "lora_te"))):
                lora_keys += 1
        # Fichier LoRA range dans le dossier checkpoints (erreur classique): le charger
        # comme transformer envoie diffusers chercher une config par defaut (SD1.5) ->
        # 404 'stable-diffusion-v1-5 does not appear to have a file named config.json'.
        if lora_keys >= 4:
            return "LoRA file, not a checkpoint - move it to the LoRA folder and pick it in Models > LoRA"
        # Encodeur texte range avec les checkpoints (telechargement Civitai 'text encoder'):
        # ce n'est pas un modele d'image, le dequantifier gaspillerait ~15 Go de cache et
        # le charger en transformer echouerait. Le pipe prend son encodeur du repo de base.
        if te_keys >= 4 and dit_keys == 0:
            return ("text encoder (Qwen2.5-VL), not an image model - the pipeline takes its "
                    "text encoder from the base repo; nothing to do with this file")
        # '*.qweight' = poids pre-quantifies (SVDQuant/Nunchaku, GPTQ-like). Signal net:
        # un checkpoint BF16/FP16 normal n'a jamais de 'qweight'.
        if has_qweight:
            return "SVDQuant/Nunchaku INT4"
    except Exception:
        pass
    return None


def _safetensors_dequant(path):
    """Renvoie le schema de quantification ComfyUI a dequantifier au chargement
    ('FP8', 'FP8 scaled' ou 'INT8 scaled'), sinon None (BF16/FP16 -> chemin normal).
    Format 'scaled' ComfyUI observe sur les checkpoints Civitai:
      X.weight (F8_E4M3 ou I8) + X.weight_scale (F32, scalaire ou par ligne [out,1])
      + X.comfy_quant (petit blob U8 descripteur, a jeter).
    NB: un bundle AIO dont SEUL l'encodeur texte est quantifie (transformer BF16)
    declenche aussi -> le loader dequant filtre le transformer et le laisse intact.
    U8 seul ne declenche pas: les blobs 'comfy_quant' sont U8 dans des fichiers sains."""
    try:
        hdr = _safetensors_header(path)
        has_fp8 = has_int = has_scale = False
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            dt = str(v.get("dtype", "")).upper()
            if dt.startswith("F8"):
                has_fp8 = True
            elif dt in ("I8", "I4", "U4", "INT8"):
                has_int = True
            if k.endswith(("weight_scale", "scale_weight")):
                has_scale = True
        if has_fp8:
            return "FP8 scaled" if has_scale else "FP8"
        if has_int and has_scale:
            return "INT8 scaled"
    except Exception:
        pass
    return None


# Marqueurs de cles du transformer Qwen-Image (layout original OU prefixe ComfyUI):
# utilises par le loader dequant pour refuser un checkpoint quantifie d'une AUTRE
# architecture (il chargerait des poids incoherents).
# Marqueurs de cles propres a Flux2Transformer2DModel (releves sur le transformer de
# FLUX.2-klein-4B: 169 tenseurs, prefixes transformer_blocks / single_transformer_blocks /
# x_embedder / context_embedder / double_stream_modulation_* / time_guidance_embed).
# Le nom _QWEN_KEY_MARKERS est conserve pour limiter la surface de conflit au merge
# depuis qwen/main -- seul le CONTENU change.
_QWEN_KEY_MARKERS = ("single_transformer_blocks.", "double_stream_modulation",
                     "x_embedder", "context_embedder")

# Prefixe ComfyUI/LDM des checkpoints diffusion single-file. diffusers 0.39 mappe
# Flux2Transformer2DModel avec une fonction IDENTITE (aucune conversion de cles):
# le state dict doit donc arriver AU LAYOUT DIFFUSERS, prefixe retire. Sinon toutes les
# cles sont "unexpected", aucun poids n'est charge, le modele reste sur 'meta' et
# dispatch_model casse sur "Cannot copy out of meta tensor; no data!".
_COMFY_PREFIX = "model.diffusion_model."


# ----------------------------------------------------------------------------
# Cache disque des transformers dequantifies (FP8/INT8 ComfyUI -> bf16), porte
# de crispz-studio. Un dequant lit et convertit tout le fichier (minutes sur
# HDD); le bf16 est ecrit UNE FOIS ici et les chargements suivants deviennent
# un single-file normal (secondes). CLE = fichier ORIGINAL (chemin+taille+
# mtime): supprimer ce cache est toujours sur, il se reconstruit a la demande.
# ----------------------------------------------------------------------------
import hashlib as _dqhash

_DQ_CACHE_CFG = str(CONFIG.get("dequant_cache", "auto") or "auto").strip()
try:
    DEQUANT_CACHE_MAX_GB = float(CONFIG.get("dequant_cache_max_gb", 60) or 0)
except Exception:
    DEQUANT_CACHE_MAX_GB = 60.0


def _file_key(path):
    """Identite stable et pas chere d'un fichier: (chemin absolu, taille, mtime)."""
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, int(st.st_mtime))


def _dequant_cache_dir():
    """Dossier du cache de dequant, cree a la demande. None = cache desactive."""
    if _DQ_CACHE_CFG.lower() in ("off", "none", "0", "false"):
        return None
    d = (os.path.join(HERE, "cache", "dequant")
         if _DQ_CACHE_CFG.lower() in ("auto", "") else _DQ_CACHE_CFG)
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except Exception as e:
        _dbg(f"dequant cache dir unavailable ({e})")
        return None


def _dequant_cache_path(src):
    """Chemin du bf16 cache pour un checkpoint source. La cle inclut taille+mtime:
    un fichier remplace (meme nom) ne reutilise jamais l'ancien cache."""
    d = _dequant_cache_dir()
    if not d:
        return None
    try:
        p, size, mtime = _file_key(src)
    except OSError:
        return None
    h = _dqhash.sha1(
        f"{p.lower()}|{size}|{mtime}|bf16-v2".encode("utf-8")).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(src))[0][:48]
    return os.path.join(d, f"{base}.{h}.safetensors")


def _dequant_cache_prune(keep=None):
    """Plafonne le cache (dequant_cache_max_gb, 0 = illimite): supprime les fichiers
    les moins recemment UTILISES (atime, sinon mtime) jusqu'a repasser sous le seuil."""
    d = _dequant_cache_dir()
    if not d or DEQUANT_CACHE_MAX_GB <= 0:
        return
    try:
        files = []
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if not f.endswith(".safetensors") or not os.path.isfile(fp):
                continue
            st = os.stat(fp)
            files.append((max(st.st_atime, st.st_mtime), st.st_size, fp))
        total = sum(s for _t, s, _p in files)
        cap = DEQUANT_CACHE_MAX_GB * 1024**3
        for _t, size, fp in sorted(files):          # plus ancien acces d'abord
            if total <= cap:
                break
            if keep and os.path.abspath(fp) == os.path.abspath(keep):
                continue
            try:
                os.remove(fp)
                total -= size
                _log(f"dequant cache: evicted {os.path.basename(fp)} "
                     f"({size / 1024**3:.1f} GB, over the {DEQUANT_CACHE_MAX_GB:.0f} GB cap)")
            except OSError as e:
                _dbg(f"dequant cache evict failed {fp}: {e}")
    except Exception as e:
        _dbg(f"dequant cache prune failed: {e}")


def _dequant_cache_store(src, sd):
    """Ecrit le state dict dequantifie dans le cache (best effort: toute erreur est
    ignoree, le chargement courant a deja le dict en memoire). Ecriture atomique via
    un .tmp renomme -> une interruption ne laisse jamais un cache tronque."""
    dst = _dequant_cache_path(src)
    if not dst:
        return
    try:
        from safetensors.torch import save_file
        t0 = time.time()
        tmp = dst + ".tmp"
        # contiguous(): safetensors refuse les vues non contigues (issues des slices
        # de dequant); clone implicite, on est deja en RAM.
        save_file({k: v.contiguous() for k, v in sd.items()}, tmp)
        os.replace(tmp, dst)
        gb = os.path.getsize(dst) / 1024**3
        _log(f"dequant cache: saved {gb:.1f} GB in {time.time() - t0:.1f}s "
             f"-> next load of this checkpoint skips the dequant")
        _dequant_cache_prune(keep=dst)
    except Exception as e:
        _log(f"dequant cache: not saved ({e})")
        try:
            os.remove(dst + ".tmp")
        except OSError:
            pass


def _hadamard_ortho(n):
    """Matrice 'regular hadamard' du ConvRot comfy-quants -- ATTENTION, ce n'est PAS
    la construction de Sylvester: la base est ce H4 precis, etendu par produits de
    Kronecker jusqu'a n (puissance de 4), puis normalise 1/sqrt(n). Orthonormee ET
    symetrique -> la reconstruction re-multiplie simplement par la meme matrice.
    (Verifie contre src/comfy_quants/formats/convrot.py; avec un Sylvester la
    correlation aux poids de base tombe a ~0 -> bruit total.)"""
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]])
    H = h4
    while H.shape[0] < n:
        H = torch.kron(H, h4)
    if H.shape[0] != n:
        raise ValueError(f"convrot groupsize {n} is not a power of 4")
    return H / (float(n) ** 0.5)


def _safetensors_comfy_prefixed(path):
    """True si le .safetensors est au layout ComfyUI ('model.diffusion_model.*').
    Lit juste l'en-tete. Un tel fichier ne peut PAS partir tel quel dans
    from_single_file (mapping identite cote diffusers, cf. _COMFY_PREFIX)."""
    try:
        return any(k.startswith(_COMFY_PREFIX)
                   for k in _safetensors_header(path) if k != "__metadata__")
    except Exception:
        return False


def _load_dequant_state_dict(path):
    """Charge en RAM un single-file ComfyUI et le rend au LAYOUT DIFFUSERS, dequantifie
    en DTYPE (bf16) tenseur par tenseur. Sert les deux cas: quantifie (FP8/INT8 'scaled')
    et simplement PREFIXE (bf16/fp16 -- rien a dequantifier, juste le prefixe a retirer):
      - bundle AIO (transformer + encodeur texte + VAE): seules les cles
        'model.diffusion_model.*' sont gardees (VAE + encodeur = repo de base);
      - X.weight (F8/I8) * X.weight_scale (scalaire ou par ligne) -> bf16;
      - blob X.comfy_quant: si 'convrot' est declare (int8_tensorwise ComfyUI), la
        rotation de Hadamard par groupes (defaut 256) est DEFAITE apres le descale --
        sans ca les poids sont un bruit total;
      - les cles de quantification (weight_scale/scale_weight, comfy_quant, marqueur
        scaled_fp8) sont consommees/jetees.
    Le dict resultant part dans from_single_file (conversion de cles diffusers comprise).
    NB VRAM/RAM: dequantifie = empreinte d'un BF16 complet; le FP8 n'economise que le
    disque/telechargement, pas la memoire."""
    from safetensors import safe_open
    t0 = time.time()
    hdr = _safetensors_header(path)
    entries = [(k, v) for k, v in hdr.items()
               if k != "__metadata__" and isinstance(v, dict)]
    # Bundle AIO: ne garder que le transformer. (Sans prefixe ComfyUI = fichier
    # transformer-only au layout original -> pas de filtre.) Methode crispz-krea2:
    # le prefixe est retire des la LECTURE, tout l'aval (scales, qcfg, garde d'archi,
    # state dict rendu) travaille donc sur des cles au layout diffusers, sans variante.
    prefix = ""
    if any(k.startswith(_COMFY_PREFIX) for k, _ in entries):
        prefix = _COMFY_PREFIX
        entries = [(k, v) for k, v in entries if k.startswith(prefix)]
    # Garde d'architecture: un checkpoint quantifie d'un AUTRE modele (cles sans
    # aucun marqueur Qwen-Image) chargerait des poids incoherents -> refus clair.
    if not any(any(m in k[len(prefix):] for m in _QWEN_KEY_MARKERS) for k, _ in entries):
        raise RuntimeError(
            f"{os.path.basename(path)}: quantized checkpoint does not look like a "
            "FLUX.2 transformer (different architecture); this build only loads "
            "FLUX.2 Klein models.")
    # Lecture SEQUENTIELLE dans l'ordre PHYSIQUE du fichier (data_offsets): un HDD
    # s'effondre en acces aleatoire, et l'ordre des cles ne suit pas celui des donnees.
    entries.sort(key=lambda kv: kv[1].get("data_offsets", [0])[0])
    raw = {}
    qcfg = {}
    # comfy-quants declare le schema soit en blobs PAR TENSEUR (X.comfy_quant),
    # soit CENTRALEMENT dans __metadata__._quantization_metadata (variante
    # StableYogi: {"layers": {"blocks...": {"format": "int8_tensorwise",
    # "convrot": true, "convrot_groupsize": 256}}}). Ignorer cette variante
    # laisse la rotation en place -> poids en bruit total (observe sur les
    # INT8 Krea 2; meme format possible ici). Les blobs par tenseur gagnent.
    try:
        qm = json.loads((hdr.get("__metadata__") or {}).get(
            "_quantization_metadata") or "{}")
        for lk, lv in (qm.get("layers") or {}).items():
            if isinstance(lv, dict):
                qcfg[lk[len(prefix):] if prefix and lk.startswith(prefix) else lk] = lv
        if qcfg:
            _dbg(f"quantization metadata: {len(qcfg)} layer(s) declared in header")
    except Exception as e:
        _dbg(f"_quantization_metadata unreadable: {e}")
    with safe_open(path, framework="pt", device="cpu") as f:
        for k, _ in entries:
            kk = k[len(prefix):]
            if kk.endswith(".comfy_quant"):  # blob JSON: format + convrot eventuels
                try:
                    qcfg[kk[:-len(".comfy_quant")]] = json.loads(
                        bytes(f.get_tensor(k).tolist()).decode("utf-8"))
                except Exception as e:
                    _dbg(f"comfy_quant blob unreadable {k}: {e}")
                continue
            raw[kk] = f.get_tensor(k)
    # Le calcul de dequantification (cast fp32 + scales + un-rotation) est limite par la
    # BANDE PASSANTE MEMOIRE en CPU (mesure crispz-krea2: ~9 min sur un INT8 12.9B): on le
    # fait sur le GPU quand il y en a un, tenseur par tenseur (quelques centaines de Mo de
    # VRAM au plus), retour bf16 en RAM. config convert_device: auto (defaut) | cpu.
    dev = "cpu"
    try:
        if (torch.cuda.is_available()
                and str(CONFIG.get("convert_device", "auto")).lower() != "cpu"):
            dev = "cuda"
    except Exception:
        pass
    _had = {}                                # cache Hadamard par taille de groupe
    sd = {}
    n_dq = n_rot = 0
    for k in list(raw.keys()):
        if (k.endswith((".weight_scale", ".scale_weight", ".scale_input", ".input_scale"))
                or k.endswith("scaled_fp8")):
            continue                         # consommees via lookup / jetees (scale_input
                                             # = echelle d'ACTIVATION, pas de poids)
        t = raw.pop(k)
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                       torch.int8, torch.uint8):
            s = None
            for cand in (k + "_scale",       # X.weight -> X.weight_scale (ComfyUI)
                         (k[:-len(".weight")] + ".scale_weight")
                         if k.endswith(".weight") else None):
                if cand and cand in raw:
                    s = raw[cand]
                    break
            t = t.to(dev).to(torch.float32)
            if s is not None:                # scalaire ou [out,1] -> broadcast
                t = t * s.to(dev).to(torch.float32)
            # ConvRot (int8_tensorwise comfy-quants): les poids stockes ont ete tournes
            # W_rot = (W.view(out, in/g, g) @ H.T).reshape(...) AVANT quantification ->
            # reconstruction = re-multiplier par H (orthonormee, symetrique) par groupe.
            cfg = qcfg.get(k[:-len(".weight")]) if k.endswith(".weight") else None
            if cfg and cfg.get("convrot"):
                g = int(cfg.get("convrot_groupsize", 256) or 256)
                if t.dim() == 2 and g > 1 and t.shape[1] % g == 0:
                    if g not in _had:
                        _had[g] = _hadamard_ortho(g).to(dev)
                    t = (t.view(t.shape[0], -1, g) @ _had[g]).reshape(t.shape[0], -1)
                    n_rot += 1
            t = t.to(DTYPE).cpu()
            n_dq += 1
        elif t.is_floating_point() and t.dtype != DTYPE:
            t = t.to(DTYPE)
        sd[k] = t
    raw.clear()
    if dev != "cpu":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    _log((f"dequantized {n_dq} tensors on {dev}" if n_dq
          else "read (already bf16/fp16, no dequant)")
         + f" ({len(sd)} kept"
         + (f", {n_rot} un-rotated (ConvRot)" if n_rot else "")
         + f") to bf16 in {time.time() - t0:.1f}s")
    return sd


# Architectures acceptees dans les .gguf. Un GGUF de diffusion declare son archi dans
# 'general.architecture': 'flux'/'flux2' (FLUX.1 comme FLUX.2 -- l'etiquette ne les
# distingue PAS), 'qwen_image', 'krea2', 'llama'/'gemma3' pour les LLM.
# On accepte donc largement au niveau de l'etiquette et on laisse le LAYOUT trancher
# (_gguf_layout): les noms de tenseurs de FLUX.2 sont une preuve, pas une declaration.
# Surchargeable par config 'gguf_arch' (chaine, ou liste separee par des virgules).
GGUF_ARCH = str(CONFIG.get("gguf_arch") or "flux2,flux").strip().lower()
GGUF_ARCHS = {a.strip() for a in GGUF_ARCH.split(",") if a.strip()}

_GGUF_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _gguf_skip(f, t):
    """Avance le flux au-dela d'une valeur GGUF sans la lire (strings et arrays inclus)."""
    import struct
    if t == 8:                                   # string
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        return
    if t == 9:                                   # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et in _GGUF_FIXED:
            f.seek(struct.calcsize(_GGUF_FIXED[et]) * n, 1)
        else:
            for _ in range(n):
                _gguf_skip(f, et)
        return
    f.seek(struct.calcsize(_GGUF_FIXED[t]), 1)


def _gguf_arch(path, max_kv=64):
    """'general.architecture' d'un .gguf -- lit seulement l'en-tete (quelques Ko), jamais
    les poids. Renvoie 'qwen_image' / 'flux' / 'krea2' / 'llama'... ou None si illisible
    (dans ce cas on ne filtre pas: mieux vaut tenter que d'ecarter un modele valide)."""
    import struct
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.seek(4 + 8, 1)                     # version (u32) + tensor_count (u64)
            nkv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(min(nkv, max_kv)):
                kl = struct.unpack("<Q", f.read(8))[0]
                if kl > 4096:                    # en-tete incoherent -> on abandonne
                    return None
                key = f.read(kl).decode("utf-8", "replace")
                t = struct.unpack("<I", f.read(4))[0]
                if key == "general.architecture" and t == 8:
                    n = struct.unpack("<Q", f.read(8))[0]
                    return f.read(n).decode("utf-8", "replace").strip().lower()
                _gguf_skip(f, t)
    except Exception as e:
        _dbg(f"gguf header read failed {path}: {e}")
    return None


# Prefixes de tenseurs du layout Qwen ORIGINAL (celui que le loader GGUF de diffusers
# sait mapper — GGUF QuantStack/city96). Certains GGUF Civitai sont convertis avec un
# schema compact renomme (blocks.N.attn.wq, txtmlp, tproj... — outil type
# stable-diffusion.cpp): l'archi declaree est bien 'qwen_image' mais AUCUNE cle ne
# matche -> tous les poids restent sur le device 'meta' et le .to(device) explose en
# "Cannot copy out of meta tensor". On detecte ce cas a l'en-tete pour refuser proprement.
_GGUF_OK_PREFIXES = ("transformer_blocks.", "single_transformer_blocks.",
                     "x_embedder", "context_embedder", "double_stream_modulation",
                     "single_stream_modulation", "time_guidance_embed",
                     "norm_out", "proj_out")

# Signature qui identifie POSITIVEMENT un transformer FLUX.2, par opposition aux autres
# DiT diffusers. Les prefixes ci-dessus sont trop laches: 'transformer_blocks.',
# 'norm_out' et 'proj_out' existent AUSSI chez Qwen-Image et FLUX.1. Ces trois cles-la,
# non -- relevees sur le transformer reel de FLUX.2-klein-4B (169 tenseurs).
# Le nom de la constante est garde pour limiter la surface de conflit au merge amont.
_QWEN_GGUF_SIGNATURE = ("x_embedder.weight", "context_embedder.weight",
                        "double_stream_modulation_img.linear.weight")


def _gguf_layout(path):
    """Etat du layout de tenseurs d'un .gguf, lu a l'en-tete (gguf mmap):
      'flux2'   -> signature FLUX.2 presente: c'est une PREUVE, bien plus fiable que
                   le 'general.architecture' declare (des outils de conversion tamponnent
                   n'importe quoi -- vu 'wan' sur des Qwen-Image parfaitement valides);
      'foreign' -> des noms lisibles, mais aucun marqueur diffusers connu (conversion
                   type stable-diffusion.cpp: blocks.N.attn.wq, txtmlp, tproj...);
      'unknown' -> en-tete illisible ou layout diffusers sans la signature FLUX.2: on ne
                   tranche pas ici, l'archi declaree reste le juge."""
    try:
        from gguf import GGUFReader
        names = [t.name for t in GGUFReader(path).tensors]
        if not names:
            return "unknown"
        if all(any(n == sig for n in names) for sig in _QWEN_GGUF_SIGNATURE):
            return "flux2"
        if any(n.startswith(_GGUF_OK_PREFIXES) for n in names):
            return "unknown"
        return "foreign"
    except Exception as e:
        _dbg(f"gguf layout check failed {path}: {e}")
        return "unknown"


def _gguf_layout_unsupported(path):
    """Renvoie une raison (str) si le .gguf n'utilise PAS le layout de tenseurs FLUX.2
    attendu par diffusers, sinon None. Lecture d'en-tete seule."""
    if _gguf_layout(path) != "foreign":
        return None
    return ("GGUF with a non-standard tensor layout (e.g. stable-diffusion.cpp "
            "conversion); diffusers cannot map it — use a QuantStack/city96-style "
            "GGUF or the BF16/FP16 .safetensors build")


def _checkpoint_dirs():
    """Dossiers a scanner pour les checkpoints single-file: principal + extra (si defini),
    sans doublon de chemin."""
    dirs = [CHECKPOINTS_DIR]
    if CHECKPOINTS_EXTRA_DIR and CHECKPOINTS_EXTRA_DIR not in dirs:
        dirs.append(CHECKPOINTS_EXTRA_DIR)
    return dirs


def list_checkpoints():
    """Modeles Qwen-Image single-file (.safetensors / .gguf) des dossiers checkpoints
    (principal + extra, fusionnes dans une seule liste). Les FP8/INT8 'scaled' ComfyUI
    sont acceptes (loader dequant, cf. _safetensors_dequant); seuls restent ecartes,
    avec leur raison: LoRA egarees, SVDQuant/Nunchaku INT4, GGUF d'une autre archi ou
    au layout sd.cpp. En cas de meme nom de fichier, le dossier principal a la
    priorite."""
    out = []
    seen = set()
    for d in _checkpoint_dirs():
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f in seen:
                continue
            if not f.lower().endswith((".safetensors", ".ckpt", ".pt", ".sft", ".gguf")):
                continue
            if f.lower().endswith(".safetensors"):
                reason = _safetensors_unsupported(os.path.join(d, f))
                if reason:
                    _log(f"checkpoint skipped ({reason}): {f}")
                    continue
            if f.lower().endswith(".gguf"):
                fp = os.path.join(d, f)
                lay, a = _gguf_layout(fp), _gguf_arch(fp)
                # Le LAYOUT prime sur l'archi declaree: les noms de tenseurs sont une
                # preuve, le KV 'general.architecture' une simple etiquette -- et des
                # outils de conversion la tamponnent faux (Qwen-Image publies en 'wan').
                if lay == "foreign":
                    _log(f"checkpoint skipped ({_gguf_layout_unsupported(fp)}): {f}")
                    continue
                if lay == "flux2":
                    if a and a not in GGUF_ARCHS:
                        _log(f"GGUF declares architecture '{a}' but its tensors ARE a "
                             f"FLUX.2 transformer (mislabelled by the conversion "
                             f"tool) -> loaded anyway: {f}")
                # layout indetermine -> l'archi declaree reste le juge.
                elif a and a not in GGUF_ARCHS:
                    _log(f"checkpoint skipped (GGUF architecture '{a}', this build only "
                         f"loads {sorted(GGUF_ARCHS)}; that model needs its own pipeline "
                         f"and text encoder/VAE): {f}")
                    continue
            seen.add(f)
            out.append(f)
    return sorted(out)


def resolve_checkpoint(name):
    """Chemin absolu d'un checkpoint single-file depuis son nom de fichier, cherche dans
    les dossiers checkpoints (principal puis extra). Renvoie name tel quel s'il est deja
    absolu; fallback sur le dossier principal si introuvable."""
    if not name or os.path.isabs(name):
        return name
    for d in _checkpoint_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(CHECKPOINTS_DIR, name)


def list_loras():
    """LoRA (.safetensors / .ckpt / .pt) du dossier loras, RECURSIF (sous-dossiers inclus).
    Renvoie des chemins RELATIFS a LORAS_DIR avec des '/' (ex. 'sous-dossier/ma_lora.safetensors')
    -> set_loras / resolve les resolvent via os.path.join(LORAS_DIR, name)."""
    exts = (".safetensors", ".ckpt", ".pt")
    out, seen = [], set()
    for d in _lora_dirs():          # principal puis extras: meme nom -> le principal gagne
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(exts):
                    rel = os.path.relpath(os.path.join(root, f), d).replace(os.sep, "/")
                    if rel.lower() not in seen:
                        seen.add(rel.lower())
                        out.append(rel)
    return sorted(out)


def set_checkpoints_dir(path):
    global CHECKPOINTS_DIR
    if path:
        CHECKPOINTS_DIR = path


def set_checkpoints_extra_dir(path):
    """Definit (ou efface avec '' / None) le dossier checkpoints supplementaire."""
    global CHECKPOINTS_EXTRA_DIR
    CHECKPOINTS_EXTRA_DIR = (path or "").strip()


def set_loras_dir(path):
    global LORAS_DIR
    if path:
        LORAS_DIR = path


def set_loras_extra_dirs(spec):
    """Definit (ou efface avec '' / [] / None) les dossiers LoRA supplementaires.
    spec = liste ou chaine 'a;b'."""
    global LORAS_EXTRA_DIRS
    LORAS_EXTRA_DIRS = _split_dirs(spec)


def _read_safetensors_metadata(path):
    """Lit le header JSON (__metadata__) d'un .safetensors SANS charger les poids."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
    return (json.loads(header.decode("utf-8")) or {}).get("__metadata__", {}) or {}


def lora_keywords(path):
    """Extrait les mots-cles / trigger words d'une LoRA depuis ses metadonnees:
    champs trigger explicites + top tags d'entrainement (ss_tag_frequency)."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        meta = _read_safetensors_metadata(path)
    except Exception as e:
        _dbg(f"lora metadata read failed: {e}")
        return ""
    words = []
    for k in ("ss_trigger_words", "modelspec.trigger_phrase", "trigger_words",
              "activation text", "ss_activation_text"):
        v = meta.get(k)
        if v:
            words.append(v if isinstance(v, str) else ", ".join(map(str, v)))
    tf = meta.get("ss_tag_frequency")
    if tf:
        try:
            d = json.loads(tf) if isinstance(tf, str) else tf
            counts = {}
            for ds in d.values():
                for tag, c in ds.items():
                    counts[tag] = counts.get(tag, 0) + int(c)
            words.extend(sorted(counts, key=counts.get, reverse=True)[:15])
        except Exception:
            pass
    seen, out = set(), []
    for w in words:
        for part in str(w).split(","):
            part = part.strip()
            if part and part.lower() not in seen:
                seen.add(part.lower())
                out.append(part)
    return ", ".join(out)


def set_loras(slots):
    """Definit les LoRA actives. slots = liste de (nom_ou_None, poids). Resout les
    noms en chemins, ignore les None.

    NE recharge PAS le modele: les LoRA sont echangees A CHAUD sur le transformer deja
    en VRAM (_apply_loras, appele par _ensure_base au run suivant)."""
    global LORAS
    new = []
    for name, weight in slots:
        if name and name not in ("None", "none", ""):
            new.append((resolve_lora_path(name), float(weight)))
    if new != LORAS:
        LORAS = new
        _log("LoRAs -> " + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in new) or "(none)")
             + " -> applied on next run (hot-swap, no model reload)")


def set_edit_loras(slots):
    """Definit les LoRA d'EDITION (pipe omni). slots = liste de (nom_ou_None, poids);
    un nom est un preset cz_edit_loras (telecharge a la demande), un chemin absolu, ou un
    fichier relatif a LORAS_DIR. Ignore les None. Applique a chaud au prochain
    generate_omni (_apply_edit_loras). Leve si un preset ne peut pas etre telecharge."""
    global EDIT_LORAS
    import cz_edit_loras
    new = []
    for name, weight in slots:
        if not name or name in ("None", "none", ""):
            continue
        name = cz_edit_loras.strip_label(name) if isinstance(name, str) else name
        if cz_edit_loras.spec(name) is not None:
            p = cz_edit_loras.resolve(name)
        else:
            p = resolve_lora_path(name)
        new.append((p, float(weight)))
    if new != EDIT_LORAS:
        EDIT_LORAS = new
        _log("edit LoRAs -> " + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in new)
                                or "(none)") + " -> applied on next edit (hot-swap)")


def edit_speed_choices():
    """Libelles du dropdown 'Edit speed' (Off, Auto, Lightning N steps...)."""
    import cz_edit_loras
    return ["Off"] + cz_edit_loras.speed_names()


def set_edit_speed(name):
    """Mode rapide de l'edition. 'Off'/None -> steps/guidance des Settings, pas de LoRA
    de vitesse. 'Auto' -> profil model_profiles du modele d'edition (Rapid-AIO / merge
    Lightning: deja distille, 4-8 steps, CFG off), sans LoRA. 'Lightning N steps' ->
    LoRA Lightning (2509 ou 2511 selon le modele d'edition; cherchee dans les dossiers
    LoRA, sinon telechargee) + N steps + guidance 1.0. Renvoie le dict applique."""
    global EDIT_SPEED
    import cz_edit_loras
    name = (name or "Off").strip()
    if name.lower() in ("off", "none", ""):
        if EDIT_SPEED is not None:
            _log("edit speed -> off (Settings steps/guidance)")
        EDIT_SPEED = None
        return None
    if name.lower().startswith("auto"):
        from cz_core import profile_for_model
        steps, guidance = profile_for_model(os.path.basename(OMNI_MODEL or ""))
        EDIT_SPEED = {"name": name, "steps": int(steps), "guidance": float(guidance),
                      "path": None}
    else:
        EDIT_SPEED = cz_edit_loras.resolve_speed(name, OMNI_MODEL)
    _log(f"edit speed -> {EDIT_SPEED['name']}: {EDIT_SPEED['steps']} steps, "
         f"cfg {EDIT_SPEED['guidance']:g}"
         + (f", LoRA {os.path.basename(EDIT_SPEED['path'])}" if EDIT_SPEED.get("path") else ""))
    return EDIT_SPEED


def set_edit_loras_enabled(on):
    """Case 'Edit LoRAs': ON = le jeu EDIT_LORAS est pose sur le pipe omni, OFF = retire
    (le jeu reste memorise). Sans rechargement."""
    global EDIT_LORAS_ENABLED
    on = bool(on)
    if on != EDIT_LORAS_ENABLED:
        EDIT_LORAS_ENABLED = on
        _log(f"edit LoRAs {'enabled' if on else 'disabled'}")


def set_omni_model(repo):
    """No-op chez klein: l'edition multi-reference est servie par le pipeline de BASE,
    il n'y a pas de modele d'edition separe a choisir. La fonction survit parce que
    cz_ui la cable sur le dropdown 'Omni model' (contrat d'API amont); elle se contente
    de reamorcer OMNI_MODEL sur BASE_REPO et de le journaliser.
    Pour changer le modele d'edition chez klein, on change le modele tout court
    (set_zimage_model / dropdown Checkpoint)."""
    global OMNI_MODEL
    OMNI_MODEL = BASE_REPO
    if (repo or "").strip() and (repo or "").strip() != BASE_REPO:
        _log(f"Omni model ignore ({repo}): klein edite avec le modele de base "
             f"({BASE_REPO}). Change le checkpoint pour changer l'editeur.")


def list_edit_models():
    """Modeles d'EDITION disponibles. Chez klein l'editeur EST le modele de base, donc
    tout checkpoint klein chargeable fait l'affaire: on renvoie la meme liste que les
    checkpoints (le dropdown 'Omni model' de cz_ui reste alimente et coherent)."""
    return list_checkpoints()


def check_omni_available():
    """Chez klein, l'edition multi-reference est NATIVE au pipeline de base: elle est
    disponible des que le modele est chargeable, sans second telechargement ni repo a
    verifier sur le Hub. Renvoie donc toujours un message pret (le contrat cz_ui veut
    une chaine markdown)."""
    return (f"**Edit ready (native):** `{BASE_REPO}` handles multi-reference editing in "
            f"the SAME pipeline as txt2img (up to 4 refs) - no second model, no extra "
            f"VRAM. Note: klein is distilled, so **negative prompts and CFG have no "
            f"effect** (see FORK.md).")


def set_offload_mode(mode):
    """Change le mode d'offload CPU. Invalide le pipe (hooks poses au chargement)."""
    global OFFLOAD_MODE
    mode = mode if mode in OFFLOAD_CHOICES else "none"
    if mode != OFFLOAD_MODE:
        OFFLOAD_MODE = mode
        free_vram()
        _log(f"offload -> {OFFLOAD_MODE}: pipeline invalidated -> will reload")


def free_vram():
    """Libere le pipeline de base + les pipelines derives et rend la VRAM
    (palier 3: unload sur inactivite ou endpoint /unload). Rechargement paresseux."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _APPLIED_LORAS
    _BASE_PIPE = None
    _DERIVED = {}
    _LOADED_KEY = None
    _APPLIED_LORAS = []      # plus de pipe -> plus d'adaptateur pose
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# Au-dela de ce cote (px) on active l'attention slicing (whole-image 2K+ -> evite le
# spill VRAM 32 Go). En-dessous (tuiles 1024, txt2img 1024/1536) -> slicing OFF =
# attention SDPA native = RAPIDE (comme ComfyUI). Reglable via config attention_slice_above.
_SLICE_ABOVE = int(CONFIG.get("attention_slice_above", 1664))

# Garde-fou: au-dela de ce cote (px), un refine "whole image" (refine_tile=0) est auto-
# tuile (tuile 1024). Defaut = le seuil de slicing: au-dela, un whole-image serait slice
# (lent: ~120s en 2K) ET risque le spill VRAM (4K -> crash). Tuiler est plus rapide ET sur.
_AUTO_TILE_ABOVE = int(CONFIG.get("auto_refine_tile_above", _SLICE_ABOVE))

# Taille de la tuile employee par cet auto-tuilage. "auto" (defaut) = calculee par
# _pick_refine_tile ; un entier fige la taille (ancien comportement : 1024).
# Mesure (RTX 5090, sortie 4096x4096, denoise 0.40, overlap 64) : le cout par pixel est
# PLAT de 768 a 1024 (1.78 / 1.83 / 1.79 us/px) et ne grimpe qu'au-dela (2.41 a 1536,
# 3.00 a 2048). Le temps suit donc la SURFACE TUILEE (n x tuile^2), pas la taille de la
# tuile. Or a 1024 la grille deborde : pas de 960 sur 4096 -> la derniere tuile est
# rabattue et recouvre la precedente sur 832px au lieu de 64, soit 1.56x la surface de
# l'image. A 896 le pas tombe juste (1.20x) -> 36.7s au lieu de 46.9s sur la meme image,
# a nombre de tuiles (25) et de coutures (8) IDENTIQUE.
# Bornes [768, 1024] : en dessous on multiplie tuiles et coutures et chaque tuile voit
# moins de contexte (le rendu derive - un arriere-plan flou se reconstruit differemment,
# verifie visuellement) ; au-dessus l'attention devient superlineaire.
_AUTO_TILE_MIN = int(CONFIG.get("auto_refine_tile_min", 768))
_AUTO_TILE_MAX = int(CONFIG.get("auto_refine_tile_max", 1024))
_AUTO_TILE_SIZE = str(CONFIG.get("auto_refine_tile", "auto")).strip().lower()


def _pick_refine_tile(w, h, overlap):
    """Tuile qui minimise la surface tuilee pour couvrir w x h (= le cout reel de la passe).

    A surface egale on garde la PLUS GRANDE tuile : moins de coutures et plus de contexte
    par tuile. Un entier dans auto_refine_tile court-circuite le calcul (taille figee)."""
    if _AUTO_TILE_SIZE not in ("auto", "", "0"):
        try:
            return round_to_multiple(int(_AUTO_TILE_SIZE))
        except ValueError:
            _log(f"config auto_refine_tile='{_AUTO_TILE_SIZE}' invalide (attendu 'auto' ou "
                 "un entier) -> calcul automatique")
    lo = max(256, _AUTO_TILE_MIN)
    hi = max(lo, _AUTO_TILE_MAX)
    ov = max(0, int(overlap))
    cands = []
    for t in range(lo, hi + 1, 32):
        step = max(16, t - ov)
        n = len(range(0, max(1, int(w)), step)) * len(range(0, max(1, int(h)), step))
        cands.append((n * t * t, -t, t))       # surface mini, puis plus grande tuile
    return min(cands)[2]

# Plafond de denoise pour le refine TUILE. En tuiles, chaque tuile est rediffusee avec le
# prompt global -> a fort denoise la diffusion reconstruit le sujet (ex: la tasse) DANS
# chaque tuile = duplications. On plafonne donc le denoise par tuile (le contenu existant
# guide alors la diffusion, facon Ultimate SD Upscale). Le refine "whole image" garde le
# denoise demande (pas de duplication possible: une seule passe sur toute la compo).
# Reglable via config refine_tile_denoise_cap (0 = pas de plafond).
_TILE_DENOISE_CAP = float(CONFIG.get("refine_tile_denoise_cap", 0.40))

# Prompt utilise pour le refine TUILE. Le prompt global decrit TOUTE la composition (pas
# la tuile) -> le passer a chaque tuile pousse la diffusion a recreer le sujet (la tasse)
# dans des tuiles qui ne sont que du fond. Par defaut on passe donc un prompt VIDE: chaque
# tuile se contente d'affiner le detail local. Valeurs config refine_tile_prompt:
#   "" (defaut) = prompt vide par tuile
#   "global"/"scene" = reutilise le prompt de la scene (ancien comportement)
#   tout autre texte = prompt generique applique a chaque tuile (ex: "high detail, sharp")
_TILE_PROMPT = str(CONFIG.get("refine_tile_prompt", ""))


def _tile_prompt(scene_prompt):
    """Prompt a utiliser par tuile selon la config (vide par defaut, anti-duplication)."""
    if _TILE_PROMPT.strip().lower() in ("global", "scene"):
        return scene_prompt or ""
    return _TILE_PROMPT


def _set_slicing(pipe, longest_side):
    """Active/desactive l'attention slicing selon le plus grand cote a traiter. Appele
    avant CHAQUE passe de diffusion (txt2img/refine/tuile/inpaint/outpaint/omni)."""
    try:
        if int(longest_side) > _SLICE_ABOVE:
            pipe.enable_attention_slicing()
        else:
            pipe.disable_attention_slicing()
    except Exception:
        pass


def _vram_str():
    """Pic VRAM PyTorch reserve / total (pour reperer la saturation -> spill RAM partagee
    Windows = lenteur extreme, et TDR/'CUDA unknown error'). Ne voit PAS la VRAM des
    autres process (ComfyUI, etc.) -> utiliser nvidia-smi pour le total reel."""
    if DEVICE != "cuda":
        return ""
    try:
        resv = torch.cuda.memory_reserved() / 1024**3
        tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f" | VRAM {resv:.1f}/{tot:.0f} Go"
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# Qwen-Image (diffusers, BF16) : un pipeline "base" txt2img qui detient les composants,
# img2img / inpaint derives via from_pipe (poids partages, pas de VRAM en double).
# ----------------------------------------------------------------------------
def _is_gguf_path(p):
    return bool(p) and str(p).lower().endswith(".gguf")


def _effective_offload(tpath=None):
    """Offload REELLEMENT applique. Un transformer GGUF quantifie ne se deplace pas sur le
    GPU via .to(cuda) ni en sequential -> seul enable_model_cpu_offload le pose sur le GPU
    pendant le forward. On force donc 'model' pour un base GGUF, quel que soit le reglage."""
    off = OFFLOAD_MODE
    t = ZIMAGE_TRANSFORMER if tpath is None else tpath
    if DEVICE == "cuda" and _is_gguf_path(t) and off != "model":
        off = "model"
    return off


def _load_transformer(path=None, base=None):
    """Charge UNIQUEMENT le transformer courant (sans le reste du pipeline):
      - GGUF quantifie -> from_single_file + GGUFQuantizationConfig (archi = repo de base)
      - single-file .safetensors -> from_single_file
      - repo HF / dossier diffusers -> sous-dossier 'transformer'
      - pas d'override -> transformer du repo de base
    Utilise au chargement complet ET pour l'echange a chaud (_swap_transformer).
    path/base: par defaut le transformer du BASE (ZIMAGE_TRANSFORMER / BASE_REPO);
    le pipe d'EDITION passe son propre fichier (GGUF, FP8 Rapid-AIO...) + son repo
    d'edition (zimage_omni_base) pour l'architecture."""
    from diffusers import Flux2Transformer2DModel
    path = ZIMAGE_TRANSFORMER if path is None else path
    base = BASE_REPO if base is None else base
    if path:
        if _is_single_file(path):
            # Garde: un fichier non chargeable (LoRA egaree, FP8, quantifie) doit
            # echouer avec un message actionnable, pas partir chercher une config
            # par defaut sur le Hub. (Sans effet sur les .gguf: header illisible -> None.)
            bad = _safetensors_unsupported(path)
            if bad:
                raise RuntimeError(f"{os.path.basename(path)}: {bad}.")
            if _is_gguf_path(path):
                # transformer Qwen GGUF (quantifie) -> tient en VRAM (~11 Go en Q4) et
                # reste rapide. Le VAE + encodeur texte viennent du repo de base (cache).
                lay = _gguf_layout_unsupported(path)
                if lay:
                    raise RuntimeError(
                        f"{os.path.basename(path)}: {lay}.")
                from diffusers import GGUFQuantizationConfig
                _log(f"loading Klein transformer (GGUF, quantized): {path} ...")
                # config/subfolder = archi du transformer depuis le repo de base (cache),
                # sinon from_single_file ne sait pas la structure et tente un repo par defaut.
                return _load_monitor(
                    f"transformer {os.path.basename(path)} (GGUF)",
                    lambda: Flux2Transformer2DModel.from_single_file(
                        path,
                        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
                        config=base, subfolder="transformer",
                        torch_dtype=DTYPE))
            dq = _safetensors_dequant(path)
            if dq:
                # FP8/INT8 'scaled' ComfyUI (builds Civitai legers) -> dequant en RAM
                # puis chargement du dict (conversion de cles diffusers incluse).
                # Deja dequantifie une fois ? -> relire le bf16 du cache
                # disque: un single-file normal (secondes) au lieu de reconvertir
                # tout le fichier (minutes sur HDD).
                cached = _dequant_cache_path(path)
                if cached and os.path.isfile(cached):
                    _log(f"loading Klein transformer ({dq} -> bf16, from dequant "
                         f"cache): {os.path.basename(cached)}")
                    try:
                        os.utime(cached, None)       # marque l'usage pour le LRU
                    except OSError:
                        pass
                    return _load_monitor(
                        f"transformer {os.path.basename(path)} (cached bf16)",
                        lambda: Flux2Transformer2DModel.from_single_file(
                            cached, config=base, subfolder="transformer",
                            torch_dtype=DTYPE))
                _log(f"loading Klein transformer (single-file, {dq} ComfyUI -> "
                     f"dequantized to bf16): {path} ...")
                sd = _load_dequant_state_dict(path)
                _dequant_cache_store(path, sd)
                return _load_monitor(
                    f"transformer {os.path.basename(path)} ({dq})",
                    lambda: Flux2Transformer2DModel.from_single_file(
                        sd, config=base, subfolder="transformer",
                        torch_dtype=DTYPE))
            # checkpoint Qwen single-file (.safetensors bf16/fp16) -> override transformer.
            # config/subfolder = archi du transformer depuis le repo de base (deja en cache),
            # comme pour le GGUF: sans ca, from_single_file ne sait pas la structure et va
            # chercher un repo par defaut -> echec en mode offline (HF_HUB_OFFLINE=1).
            if _safetensors_comfy_prefixed(path):
                # Layout ComfyUI SANS quantification: diffusers ne convertit pas les cles
                # Qwen (mapping identite), un passage direct du chemin laisserait le modele
                # sur 'meta'. On lit et on deprefixe nous-memes -- meme cout RAM, puisque
                # from_single_file charge de toute facon tout le checkpoint. Pas de cache
                # disque ici: il n'y a rien de dequantifie a memoriser, ce serait une
                # copie bf16 -> bf16.
                _log(f"loading Klein transformer (single-file, ComfyUI layout -> "
                     f"diffusers): {path} ...")
                sd = _load_dequant_state_dict(path)
                return _load_monitor(
                    f"transformer {os.path.basename(path)}",
                    lambda: Flux2Transformer2DModel.from_single_file(
                        sd, config=base, subfolder="transformer",
                        torch_dtype=DTYPE))
            _log(f"loading Klein transformer (single-file): {path} ...")
            return _load_monitor(
                f"transformer {os.path.basename(path)}",
                lambda: Flux2Transformer2DModel.from_single_file(
                    path, config=base, subfolder="transformer",
                    torch_dtype=DTYPE))
        # repo HF / dossier diffusers -> charge le sous-dossier 'transformer'.
        _log(f"loading Klein transformer (repo subfolder): {path} ...")
        return _load_monitor(
            f"transformer {path}",
            lambda: Flux2Transformer2DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=DTYPE))
    _log(f"loading Klein transformer (base repo): {base} ...")
    return _load_monitor(
        f"transformer {base}",
        lambda: Flux2Transformer2DModel.from_pretrained(
            base, subfolder="transformer", torch_dtype=DTYPE))


def _lora_names(loras):
    return [f"cz_lora_{i}" for i in range(len(loras))]


def _clear_loras(pipe):
    """Retire TOUT adaptateur LoRA du pipe pour repartir d'un etat vierge.

    unload_lora_weights() seul laisse, selon les versions diffusers/peft, un peft_config
    residuel sur le transformer -> le load suivant avertit ('Already found a peft_config')
    et, comme on reutilise les memes noms d'adaptateurs (cz_lora_i), l'ancien adaptateur
    peut rester en place (mauvaise LoRA appliquee). On supprime donc explicitement les
    adaptateurs restants par nom apres l'unload."""
    try:
        pipe.unload_lora_weights()
    except Exception as e:
        _dbg(f"unload_lora_weights: {e}")
    try:
        listed = pipe.get_list_adapters() or {}
        names = sorted({n for lst in listed.values() for n in (lst or [])})
        if names:
            pipe.delete_adapters(names)
            _dbg(f"cleared leftover LoRA adapters: {names}")
    except Exception as e:
        _dbg(f"delete_adapters: {e}")


def _sync_adapters(pipe, wanted, applied, force=False, tag="LoRA"):
    """Synchronise les adaptateurs PEFT d'un pipe avec le jeu `wanted`, SANS recharger
    le modele. `applied` = jeu reellement pose sur ce pipe (liste de (chemin, poids)).

    Le transformer reste en VRAM; seuls les adaptateurs PEFT bougent:
      - memes fichiers, poids differents -> set_adapters (immediat)
      - jeu de LoRA different            -> unload_lora_weights + reload des LoRA (~1s)
    Les pipes derives (from_pipe) partagent ce transformer -> ils suivent automatiquement.
    Renvoie (ok, applied): ok=False si echec (le caller decide: reload complet pour le
    base, erreur franche pour l'edition), applied = nouveau jeu pose ([] si echec)."""
    wanted = list(wanted)
    if not force and applied == wanted:
        return True, applied
    old_paths = [p for p, _ in applied]
    new_paths = [p for p, _ in wanted]
    try:
        if not force and old_paths and old_paths == new_paths:
            # Seuls les poids changent -> re-ponderation instantanee.
            pipe.set_adapters(_lora_names(wanted), [float(w) for _, w in wanted])
            _log(f"{tag} weights updated in place (no reload): "
                 + ", ".join(f"{os.path.basename(p)}@{w}" for p, w in wanted))
            return True, wanted
        if old_paths or force:
            _clear_loras(pipe)
        names, weights = [], []
        for i, (p, w) in enumerate(wanted):
            if os.path.isfile(p):
                an = f"cz_lora_{i}"
                _log(f"applying {tag}: {os.path.basename(p)} (weight {w})")
                # Passer le dossier + weight_name (et non le chemin complet) : sinon
                # diffusers en mode offline (HF_HUB_OFFLINE) refuse "must specify a
                # weight_name". Marche aussi online et avec un fichier local direct.
                # A partir du 2e adaptateur, peft avertit "Already found a peft_config"
                # : empiler plusieurs LoRA est justement le but, on tait ce message.
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Already found a `peft_config`.*")
                    pipe.load_lora_weights(os.path.dirname(p) or ".",
                                           weight_name=os.path.basename(p), adapter_name=an)
                names.append(an)
                weights.append(float(w))
            else:
                _log(f"{tag} file not found, ignored: {p}")
        if names:
            pipe.set_adapters(names, weights)
        if not force:
            _log(f"{tag}s hot-swapped (no model reload)")
        return True, wanted
    except Exception as e:
        _log(f"{tag} hot-swap failed ({e})")
        return False, []


def _apply_loras(pipe, force=False):
    """LoRA du BASE (txt2img/img2img): synchronise le pipe avec LORAS via _sync_adapters.
    Renvoie True si applique, False si echec (le caller retombe sur un reload complet)."""
    global _APPLIED_LORAS
    ok, _APPLIED_LORAS = _sync_adapters(pipe, LORAS, _APPLIED_LORAS, force=force)
    if not ok:
        _log("falling back to a full reload")
    return ok


def _apply_edit_loras(pipe):
    """LoRA d'EDITION: synchronise le pipe omni avec EDIT_LORAS (ou [] si la case
    'Edit LoRAs' est decochee). Un echec est une erreur franche: l'utilisateur a
    demande ce preset, une edition SANS lui serait un faux resultat."""
    global _APPLIED_EDIT_LORAS
    wanted = list(EDIT_LORAS) if EDIT_LORAS_ENABLED else []
    # LoRA Lightning du mode rapide: empilee APRES les presets (independante de la
    # case 'Edit LoRAs', qui ne concerne que les presets de tache).
    if EDIT_SPEED and EDIT_SPEED.get("path"):
        wanted.append((EDIT_SPEED["path"], 1.0))
    ok, _APPLIED_EDIT_LORAS = _sync_adapters(pipe, wanted, _APPLIED_EDIT_LORAS,
                                             tag="edit LoRA")
    if not ok:
        raise RuntimeError("edit LoRA could not be applied on the edit pipe "
                           "(see log); nothing was generated")


def _swap_transformer(pipe):
    """Remplace SEULEMENT le transformer du pipeline deja en cache: le VAE, l'encodeur de
    texte, le tokenizer et le scheduler restent en VRAM (c'est eux le gros du temps de
    chargement). Valable uniquement a repo de base + offload EFFECTIF identiques.

    Renvoie True si l'echange a reussi, False -> le caller fait un reload complet."""
    global _APPLIED_LORAS, _DERIVED
    t0 = time.time()
    old_t = _LOADED_KEY[1] if _LOADED_KEY else None
    # Passer de/vers un GGUF change l'offload EFFECTIF (un GGUF impose 'model') -> les
    # hooks accelerate et le placement different: on ne bricole pas, on recharge.
    if _effective_offload(old_t) != _effective_offload(ZIMAGE_TRANSFORMER):
        _log("transformer swap skipped (GGUF changes the effective offload) -> full reload")
        return False
    try:
        _log(f"switching Qwen transformer -> {ZIMAGE_TRANSFORMER or BASE_REPO} "
             "(keeping VAE + text encoder in VRAM)")
        new_t = _load_transformer()
        old = getattr(pipe, "transformer", None)
        off = _effective_offload()
        # Offload: les hooks accelerate sont poses sur les composants. Il faut les retirer
        # avant l'echange, sinon le nouveau transformer n'en a pas et l'ancien garde les siens.
        if DEVICE == "cuda" and off in ("model", "sequential"):
            try:
                pipe.remove_all_hooks()
            except Exception as e:
                _dbg(f"remove_all_hooks: {e}")
        try:
            pipe.register_modules(transformer=new_t)   # API diffusers (met a jour le config)
        except Exception:
            pipe.transformer = new_t
        # Liberer l'ANCIEN transformer AVANT de poser le nouveau sur le GPU: sinon
        # ancien + nouveau + VAE/encodeur depassent la VRAM -> spill en RAM partagee
        # qui ne se resorbe pas (mesure sur une grille XYZ multi-checkpoints cote
        # studio: 1.7 s/step -> 300-600 s/step, puis crash). Les pipes derives
        # (from_pipe) pointent aussi sur l'ancien -> a purger d'abord, sinon
        # `del old` ne libere rien (from_pipe est gratuit, il sera reconstruit).
        _DERIVED = {}
        del old
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if DEVICE == "cuda":
            if off == "model":
                pipe.enable_model_cpu_offload()
            elif off == "sequential":
                pipe.enable_sequential_cpu_offload()
            else:
                new_t.to(DEVICE)       # jamais un GGUF ici (offload force a 'model')
        # Les adaptateurs LoRA etaient poses sur l'ancien transformer -> a reposer.
        _APPLIED_LORAS = []
        if LORAS:
            _apply_loras(pipe, force=True)
        _log(f"transformer switched in {time.time() - t0:.1f}s "
             "(VAE + text encoder kept, no full reload)")
        return True
    except Exception as e:
        _log(f"transformer hot-swap failed ({e}); falling back to a full reload")
        _APPLIED_LORAS = []
        return False


def _ensure_base():
    """Charge (si besoin) le pipeline de base txt2img. Gere le transformer
    single-file/GGUF et l'offload. Cache par (repo, transformer, offload).

    Deux echanges a chaud evitent un rechargement complet (transformer + VAE + encodeur
    texte, des dizaines de secondes):
      - LoRA differentes            -> _apply_loras (adaptateurs PEFT seuls)
      - transformer different, meme repo de base + offload -> _swap_transformer."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _BASE_SCHED_CONFIG, _APPLIED_LORAS
    key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE)
    _dbg(f"_ensure_base key={key} cached={_LOADED_KEY}")
    if _BASE_PIPE is not None and _LOADED_KEY == key:
        if _apply_loras(_BASE_PIPE):
            _dbg("base pipeline: reusing cached (no reload)")
            return _BASE_PIPE
        _dbg("base pipeline: LoRA hot-swap failed -> free + reload")
        free_vram()
    elif _BASE_PIPE is not None:
        # Seul le transformer change (meme repo de base + meme offload) ? -> on ne recharge
        # QUE le transformer et on garde VAE + encodeur texte en VRAM.
        if (_LOADED_KEY and _LOADED_KEY[0] == BASE_REPO and _LOADED_KEY[2] == OFFLOAD_MODE
                and _swap_transformer(_BASE_PIPE)):
            _LOADED_KEY = key
            return _BASE_PIPE
        _dbg("base pipeline: key changed -> free + reload")
        free_vram()
    from diffusers import Flux2KleinPipeline
    t0 = time.time()
    kwargs = {}
    if ZIMAGE_TRANSFORMER:
        kwargs["transformer"] = _load_transformer()
    _log(f"loading FLUX.2 Klein base: {BASE_REPO} (offload={OFFLOAD_MODE}, dtype=bf16) ... "
         "first time downloads from HF (~15 Go), then cached")
    pipe = _load_monitor(f"FLUX.2 Klein base {BASE_REPO}",  # noqa: E128
                         lambda: Flux2KleinPipeline.from_pretrained(BASE_REPO, torch_dtype=DTYPE,
                                                                    **kwargs))
    # Capture le config natif (flow-matching) du scheduler -> base pour construire les
    # autres samplers (euler/dpm2a/dpmpp2m) sans perdre shift/flow params.
    try:
        _BASE_SCHED_CONFIG = dict(pipe.scheduler.config)
    except Exception:
        _BASE_SCHED_CONFIG = None
    # LoRA Qwen-Image (sur le transformer du base -> partage par les pipes derives).
    # force=True: pipe neuf, aucun adaptateur pose -> on (re)pose tout.
    _APPLIED_LORAS = []
    if LORAS:
        _apply_loras(pipe, force=True)
    # Attention slicing: POSE PAR APPEL via _set_slicing (selon la resolution traitee),
    # PAS au chargement. En tuile/1024 -> slicing OFF = attention SDPA native, rapide
    # (comme ComfyUI). Whole-image 2K+ -> slicing ON pour eviter le spill VRAM 32 Go.
    # enable_*_cpu_offload gere lui-meme le device -> ne PAS faire .to(cuda) alors.
    # IMPORTANT: un transformer GGUF quantifie ne se deplace PAS sur le GPU via .to(cuda)
    # (offload=none) ni en sequential -> il reste sur CPU = ULTRA lent (VRAM vide, ~500s/step).
    # Seul enable_model_cpu_offload (accelerate) le pose correctement sur le GPU pendant le
    # forward. On force donc 'model' pour un base GGUF, quel que soit le reglage UI/config.
    _off = _effective_offload()
    if _off != OFFLOAD_MODE:
        _log(f"GGUF base: offload '{OFFLOAD_MODE}' force a '{_off}' (un GGUF ne tourne pas "
             f"sur GPU en none/sequential -> sinon CPU, ~500s/step)")
    if DEVICE == "cuda" and _off == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and _off == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    # VAE tiling/slicing: indispensable pour l'img2img/upscale. Qwen-Image est gros (~20B
    # transformer + encodeur texte) -> sans tiling le VAE peut faire deborder la VRAM (spill
    # RAM partagee = tres lent). Tuiler le VAE plafonne ce pic (comme le "tiled decode" de
    # ComfyUI). Le VAE est partage par les pipes derives.
    try:
        pipe.vae.config.force_upcast = False   # VAE en bf16 (fp32 lent sur Blackwell) -- TOUJOURS
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception as e:
        _dbg(f"VAE tiling not available: {e}")
    _apply_sampler(pipe)   # pose le sampler choisi (euler par defaut) sur le pipe de base
    _BASE_PIPE = pipe
    _DERIVED = {"txt2img": pipe}
    _LOADED_KEY = key
    _log(f"FLUX.2 Klein base ready in {time.time() - t0:.1f}s (sampler={SAMPLER}/{SCHEDULE})")
    return pipe


def get_pipe(kind="img2img"):
    """Renvoie le pipeline demande. txt2img/img2img/inpaint derivent du base via
    from_pipe (poids partages). Omni a besoin de composants en plus (SigLIP) ->
    charge separement depuis un modele Omni dedie (CONFIG['zimage_omni_model'])."""
    if kind == "omni":
        # klein: l'edition multi-reference EST le pipeline de base (`image` accepte une
        # liste de PIL). Aucun second modele a charger -> pas de VRAM en double.
        _dbg("get_pipe('omni'): pipeline de base (multi-reference natif)")
        return _ensure_base()
    base = _ensure_base()
    if kind in _DERIVED:
        _dbg(f"get_pipe('{kind}'): reuse derived")
        return _DERIVED[kind]
    # `Flux2KleinPipeline` n'expose PAS `strength`: son `image` est un conditionnement de
    # reference (facon Kontext), pas un depart bruite. L'img2img passe donc par le pipeline
    # d'INPAINT avec un masque blanc plein (injecte par _qwen_call). Un seul objet derive
    # sert les deux -> on le partage sous les deux clefs de cache.
    from diffusers import Flux2KleinInpaintPipeline
    cls = {"img2img": Flux2KleinInpaintPipeline, "inpaint": Flux2KleinInpaintPipeline}.get(kind)
    if cls is None:
        return base
    twin = "inpaint" if kind == "img2img" else "img2img"
    if twin in _DERIVED:
        _dbg(f"get_pipe('{kind}'): reuse '{twin}' (meme pipeline Flux2KleinInpaint)")
        _DERIVED[kind] = _DERIVED[twin]
        return _DERIVED[kind]
    _log(f"deriving {kind} pipeline (shared weights, no extra VRAM)")
    # Un transformer GGUF est QUANTIFIE: on ne peut pas le recaster en dtype (.to(DTYPE)
    # leve "Casting a quantized model is unsupported"). On saute donc le recast bf16 dans
    # ce cas (le compute_dtype est deja bf16). Sinon (bf16 plein): recast defensif Blackwell
    # (certains from_pipe upcastent en float32 -> tres lent sans tensor cores fp32).
    quantized = bool(ZIMAGE_TRANSFORMER) and ZIMAGE_TRANSFORMER.lower().endswith(".gguf")
    try:
        # GGUF quantifie: torch_dtype=None EXPLICITE -> sinon from_pipe met float32 par
        # defaut et caste le modele quantifie -> ValueError "Casting a quantized model".
        p = cls.from_pipe(base, torch_dtype=None) if quantized else cls.from_pipe(base, torch_dtype=DTYPE)
    except TypeError:
        p = cls.from_pipe(base)
    try:
        if not quantized:
            p = p.to(DTYPE)
        p.vae.config.force_upcast = False
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        _log(f"img2img bf16 recast failed ({e})")
    _apply_sampler(p)   # meme sampler que le base (au cas ou from_pipe recree le scheduler)
    # Diagnostic vitesse: si le pipe derive n'est PAS sur cuda -> img2img/refine tourne
    # sur CPU = ultra lent. On le force sur DEVICE en mode plein VRAM (offload gere seul).
    # NB: offload EFFECTIF (un base GGUF force 'model' meme si l'UI dit 'none'): en
    # offload, un transformer "sur CPU" est normal -> un .to(cuda) casserait les hooks.
    try:
        tdev = next(p.transformer.parameters()).device
        if DEVICE == "cuda" and _effective_offload() == "none" and tdev.type != "cuda":
            _log(f"{kind} pipeline was on {tdev} -> moving to {DEVICE}")
            p = p.to(DEVICE)
            tdev = next(p.transformer.parameters()).device
        _log(f"{kind} pipeline ready: transformer={tdev}")
    except Exception as e:
        _dbg(f"device check failed: {e}")
    _DERIVED[kind] = p
    return p


def generate_omni(refs, prompt, negative, width, height, steps, seed,
                  guidance=None, honor_size=False, steps_explicit=False):
    """Edition multi-reference FLUX.2 Klein: edite une (ou plusieurs, jusqu'a 4) image(s)
    d'entree selon le prompt d'instruction. Conserve la signature de l'upstream (cz_ui).

    Chez klein c'est le pipeline de BASE qui edite (`image` accepte une liste de PIL):
    pas de second modele, pas de VRAM en double, pas de temps de chargement supplementaire.
    `negative` est accepte pour compat mais SANS EFFET (modele distille, cf. _cfg) -
    cz_protocol l'annonce via supports.negative = False.
    width/height sont ignores par defaut (l'edition preserve les dimensions de l'entree);
    honor_size=True les transmet au pipe. Les LoRA d'edition (EDIT_LORAS, case
    'Edit LoRAs') sont posees a chaud ici, sur le meme transformer que le base."""
    refs = [r.convert("RGB") for r in (refs or []) if r is not None]
    if not refs:
        raise ValueError("Edit needs at least one input image.")
    pipe = get_pipe("omni")
    _apply_edit_loras(pipe)
    # Mode rapide: ses steps/guidance priment sur les Settings; un appelant qui les
    # a fixes explicitement (protocole spec.steps / spec.guidance) garde la main.
    if EDIT_SPEED:
        if not steps_explicit:
            steps = EDIT_SPEED["steps"]
        if guidance is None:
            guidance = EDIT_SPEED["guidance"]
    g = float(GUIDANCE) if guidance is None else float(guidance)
    lora_info = ", edit LoRA " + "+".join(os.path.basename(p) for p, _ in _APPLIED_EDIT_LORAS) \
        if _APPLIED_EDIT_LORAS else ""            # presets + LoRA Lightning reellement poses
    _log(f"edit: {len(refs)} image(s), {int(steps)} steps, cfg {g:.1f}{lora_info} ...")
    _progress(0.1, f"Editing ({len(refs)} image(s))...")
    _set_slicing(pipe, max(max(r.size) for r in refs))
    t0 = time.time()
    # Flux2KleinPipeline accepte `list[PIL] | PIL`: on passe la liste telle quelle des
    # qu'il y a plus d'une reference.
    image_arg = refs if len(refs) > 1 else refs[0]
    size_kw = {}
    if honor_size and width and height:
        size_kw = {"width": round_to_multiple(int(width)), "height": round_to_multiple(int(height))}
        _dbg(f"edit: explicit output size {size_kw['width']}x{size_kw['height']}")
    out = _qwen_call(
        pipe,
        image=image_arg,
        prompt=prompt or "",
        num_inference_steps=int(steps),
        generator=_make_generator(seed),
        **size_kw,
        **_cfg(negative, guidance),
    ).images[0]
    _log(f"edit done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def load_pipe():
    """Compat: pipeline img2img (etage de raffinement)."""
    return get_pipe("img2img")


@_gpu_serial
def generate(prompt, width, height, steps, seed, negative_prompt=""):
    """txt2img Qwen-Image: genere une image depuis un prompt. CFG reel via true_cfg_scale
    (= curseur guidance, ~4.0), ~30-50 steps conseilles. Le negative prompt agit grace au
    vrai CFG (cf. _cfg)."""
    pipe = get_pipe("txt2img")
    w = round_to_multiple(int(width))
    h = round_to_multiple(int(height))
    _log(f"txt2img: {w}x{h}, {int(steps)} steps, cfg {GUIDANCE:.1f} ...")
    _dbg(f"txt2img seed={seed} dtype=bf16 device={DEVICE} offload={OFFLOAD_MODE} "
         f"transformer={'single-file' if ZIMAGE_TRANSFORMER else 'repo'}")
    if DEVICE == "cuda":
        _dbg(f"VRAM before: alloc={torch.cuda.memory_allocated()/1024**3:.2f} Go")
    _progress(0.1, f"Generating {w}x{h} ({int(steps)} steps)...")
    _set_slicing(pipe, max(w, h))
    t0 = time.time()
    img = _qwen_call(
        pipe,
        prompt=prompt or "",
        width=w, height=h,
        num_inference_steps=int(steps),
        generator=_make_generator(seed),
        **_cfg(negative_prompt),
    ).images[0]
    _log(f"txt2img done in {time.time() - t0:.1f}s")
    if DEVICE == "cuda":
        _dbg(f"VRAM peak: alloc={torch.cuda.max_memory_allocated()/1024**3:.2f} Go | "
             f"reserved={torch.cuda.max_memory_reserved()/1024**3:.2f} Go")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return img


def round_to_multiple(x, m=16):
    return max(m, int(round(x / m) * m))


def set_force_ratio(spec):
    """Definit le ratio force pour upscale/img2img: 'W:H' / 'WxH' (ex '13:19', '832x1216')
    ou '' pour desactiver (ratio natif preserve). Pilote par le radio UI."""
    global FORCE_RATIO
    FORCE_RATIO = (spec or "").strip()
    _log(f"force ratio -> {FORCE_RATIO or '(off, ratio natif preserve)'}")


def set_force_ratio_mode(mode):
    """'crop' (recadrage centre) ou 'extend' (outpaint des bandes manquantes)."""
    global FORCE_RATIO_MODE
    FORCE_RATIO_MODE = "extend" if str(mode or "").strip().lower() == "extend" else "crop"
    _log(f"force ratio mode -> {FORCE_RATIO_MODE}")


def _parse_ratio(spec):
    """(w, h) depuis 'W:H', 'WxH', ou un label '832 x 1216 | 13:19'; sinon None."""
    import re
    if not spec:
        return None
    m = re.search(r"(\d+)\s*[:xX×]\s*(\d+)", str(spec))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if a > 0 and b > 0 else None


def _crop_to_ratio(image, ratio_w, ratio_h):
    """Recadre (centre) l'image au ratio ratio_w:ratio_h en gardant l'aire maximale."""
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur > target:                       # trop large -> couper les cotes
        nw = max(1, int(round(h * target)))
        x0 = (w - nw) // 2
        return image.crop((x0, 0, x0 + nw, h))
    nh = max(1, int(round(w / target)))    # trop haut -> couper haut/bas
    y0 = (h - nh) // 2
    return image.crop((0, y0, w, y0 + nh))


def _extend_to_ratio(image, ratio_w, ratio_h, prompt, steps, seed):
    """Amene l'image au ratio cible en l'ETENDANT (outpaint) au lieu de recadrer:
    bandes symetriques ajoutees sur l'axe manquant et remplies par le modele via
    outpaint_directions -- le centre garde sa pleine resolution (seules les bandes
    sont generees, diffusion bornee a ~1 MP puis recomposition).

    Anti 'effet bande': une passe img2img legere (EXTEND_DENOISE) tourne sur l'image
    etendue, mais SEULES les bandes + une marge de transition feather sont recollees
    depuis cette passe -- le centre original reste PIXEL POUR PIXEL intact (la passe
    harmonise l'exposition/texture aux jointures sans jamais retoucher l'image)."""
    from PIL import ImageDraw, ImageFilter
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur < target:                       # trop etroit -> elargir gauche + droite
        pad = target * h - w
        out = outpaint_directions(image, None, ["left", "right"], prompt, steps, seed,
                                  expand=pad / (2.0 * w))
    else:                                  # trop large -> etendre haut + bas
        pad = w / target - h
        out = outpaint_directions(image, None, ["top", "bottom"], prompt, steps, seed,
                                  expand=pad / (2.0 * h))
    if EXTEND_DENOISE > 0.001:
        _log(f"extend: seam-blend pass (img2img denoise {EXTEND_DENOISE:.2f}, "
             "original centre kept)")
        refined = _refine_whole(get_pipe("img2img"), out, EXTEND_DENOISE,
                                steps, prompt, seed)
        # Masque de recollage: blanc = prendre la passe harmonisee (bandes + marge de
        # transition A CHEVAL sur la jointure), noir = garder l'original. La marge
        # penetre dans l'image d'origine puis est feather -> raccord fondu, centre intact.
        ox, oy = (out.width - w) // 2, (out.height - h) // 2
        m = max(24, int(0.05 * min(out.size)))       # transition ~5% du petit cote
        mx, my = (m if ox > 0 else 0), (m if oy > 0 else 0)   # marge cote jointure SEULEMENT
        mask = Image.new("L", out.size, 255)
        ImageDraw.Draw(mask).rectangle(
            [ox + mx, oy + my, ox + w - mx, oy + h - my], fill=0)
        mask = mask.filter(ImageFilter.GaussianBlur(max(8, m // 3)))
        out = Image.composite(refined, out, mask)
    return out


def _reframe_canvas(image, ratio_w, ratio_h, overlap=8):
    """Place l'image dans un canevas plus grand au ratio cible (expansion sur 1 axe),
    + un masque (blanc = a remplir, noir = a garder, avec un petit overlap)."""
    from PIL import ImageDraw
    image = image.convert("RGB")
    w, h = image.size
    r = ratio_w / ratio_h
    # Alignement sur 32 (patch 2 x VAE 16): evite les erreurs de conv (no engine).
    if w / h < r:  # trop etroit -> elargir
        nw, nh = round_to_multiple(int(round(h * r)), 32), round_to_multiple(h, 32)
    else:          # trop large -> agrandir en hauteur
        nw, nh = round_to_multiple(w, 32), round_to_multiple(int(round(w / r)), 32)
    nw, nh = max(nw, round_to_multiple(w, 32)), max(nh, round_to_multiple(h, 32))
    ox, oy = (nw - w) // 2, (nh - h) // 2
    canvas = Image.new("RGB", (nw, nh), (127, 127, 127))
    canvas.paste(image, (ox, oy))
    mask = Image.new("L", (nw, nh), 255)
    ImageDraw.Draw(mask).rectangle(
        [ox + overlap, oy + overlap, ox + w - overlap, oy + h - overlap], fill=0)
    return canvas, mask, nw, nh


@_gpu_serial
def inpaint_run(background, mask, prompt, steps, denoise, seed):
    """Inpaint: regenere la zone blanche du masque selon le prompt
    (ZImageInpaintPipeline). background + mask = PIL (L: blanc = a changer)."""
    orig = background.convert("RGB")
    full_mask = mask
    # Diffusion bornee a ~1 MP (zone optimale du modele), puis recomposition pleine res.
    bg, work_mask, orig_size = _cap_work_res(orig, mask)
    w, h = bg.size
    pipe = get_pipe("inpaint")
    _log(f"inpaint: work {w}x{h} (orig {orig_size[0]}x{orig_size[1]}), {int(steps)} steps, "
         f"strength {float(denoise):.2f}, cfg {GUIDANCE:.1f} ...")
    _progress(0.1, "Inpainting...")
    _set_slicing(pipe, max(w, h))
    t0 = time.time()
    out = _qwen_call(pipe, prompt=prompt or "", image=bg, mask_image=work_mask,
                     strength=float(denoise), num_inference_steps=int(steps),
                     generator=_make_generator(seed), **_cfg(None)).images[0]
    # Recompose: hors-masque garde la pleine resolution; jointure fondue (feather).
    out = _composite_back(out, orig, full_mask, orig_size,
                          feather=max(2, int(min(orig_size) * 0.01)))
    _log(f"inpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# Resolution cible "zone optimale" du modele Z-Image (~1 MP, comme les ratios txt2img).
# Le reframe vise ce budget pour ne PAS exploser le nombre de pixels (sortie 2-3 MP qui
# sort de la zone d'entrainement -> lent et qualite degradee).
MODEL_TARGET_PX = 1024 * 1024


def _ratio_canvas(ratio_w, ratio_h, target_px=MODEL_TARGET_PX):
    """Dimensions (multiples de 32) d'un canevas au ratio donne, a ~target_px pixels."""
    r = float(ratio_w) / float(ratio_h)
    nh = (target_px / r) ** 0.5
    nw = nh * r
    return round_to_multiple(int(round(nw)), 32), round_to_multiple(int(round(nh)), 32)


def _cap_work_res(image, mask, max_px=MODEL_TARGET_PX):
    """Borne la resolution de travail pour la diffusion: si image > max_px, renvoie une
    version reduite (multiples de 32) de (image, mask) + la taille d'origine pour
    recomposer ensuite. Evite de faire tourner le modele tres au-dessus de sa zone
    optimale (~1 MP) -> plus rapide et meilleure qualite."""
    w, h = image.size
    if w * h > max_px:
        s = (max_px / (w * h)) ** 0.5
        ww, wh = round_to_multiple(int(w * s), 32), round_to_multiple(int(h * s), 32)
    else:
        ww, wh = round_to_multiple(w, 32), round_to_multiple(h, 32)
    img_w = image.resize((ww, wh), Image.LANCZOS) if (ww, wh) != image.size else image
    msk_w = mask.resize((ww, wh), Image.NEAREST) if mask.size != (ww, wh) else mask
    return img_w, msk_w, (w, h)


def _composite_back(result, original, mask, orig_size, feather=0):
    """Recompose a la resolution d'origine: la zone masquee (blanc) vient de `result`
    (re-agrandi a orig_size), le reste vient de `original` -> le hors-masque garde la
    pleine resolution de l'image de depart. `feather` (px) floute le masque pour fondre
    la jointure (transition progressive original <-> genere, plus de ligne dure)."""
    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)
    if original.size != orig_size:
        original = original.resize(orig_size, Image.LANCZOS)
    m = (mask.resize(orig_size, Image.NEAREST) if mask.size != orig_size else mask).convert("L")
    if feather and feather > 0:
        from PIL import ImageFilter
        m = m.filter(ImageFilter.GaussianBlur(float(feather)))
    return Image.composite(result, original.convert("RGB"), m)


def reframe(image, ratio_w, ratio_h, fit, prompt, steps, seed, strength=1.0):
    """Recadre l'image au ratio cible en bornant la sortie a la resolution optimale du
    modele (~1 MP) -> plus d'explosion du nombre de pixels.
      fit='contain' : l'image entiere rentre dans le canevas (sans l'agrandir), les bords
                      ajoutes sont remplis par Z-Image (outpaint).
      fit='cover'   : l'image remplit le canevas au ratio puis est recadree au centre
                      (pas d'outpaint, simple reframe/crop)."""
    from PIL import ImageDraw
    img = image.convert("RGB")
    w, h = img.size
    nw, nh = _ratio_canvas(ratio_w, ratio_h)
    if str(fit).lower() == "cover":
        scale = max(nw / w, nh / h)
        rw2, rh2 = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((rw2, rh2), Image.LANCZOS)
        left, top = (rw2 - nw) // 2, (rh2 - nh) // 2
        out = resized.crop((left, top, left + nw, top + nh))
        _log(f"reframe cover: {w}x{h} -> {nw}x{nh} (crop, no fill)")
        return out
    # contain -> on adapte l'original sans l'agrandir, puis on outpaint les bords.
    from PIL import ImageFilter
    scale = min(nw / w, nh / h, 1.0)
    rw2, rh2 = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((rw2, rh2), Image.LANCZOS) if (rw2, rh2) != (w, h) else img
    ox, oy = (nw - rw2) // 2, (nh - rh2) // 2
    # Bords = extension floue des couleurs du bord (blurred edge fill, comme l'outpaint)
    # plutot qu'un gris -> continuite d'exposition; transparait si strength < 1.0.
    arr = np.pad(np.array(resized), [[oy, nh - rh2 - oy], [ox, nw - rw2 - ox], [0, 0]],
                 mode="edge")
    canvas = Image.fromarray(np.ascontiguousarray(arr))
    overlap = 8
    mask = Image.new("L", (nw, nh), 255)
    ImageDraw.Draw(mask).rectangle(
        [ox + overlap, oy + overlap, ox + rw2 - overlap, oy + rh2 - overlap], fill=0)
    blur_r = max(8, int(min(nw, nh) * 0.03))
    canvas = Image.composite(canvas.filter(ImageFilter.GaussianBlur(blur_r)), canvas, mask)
    pipe = get_pipe("inpaint")
    _log(f"reframe contain (outpaint): {w}x{h} -> {nw}x{nh}, {int(steps)} steps, "
         f"strength {float(strength):.2f}, cfg {GUIDANCE:.1f} ...")
    _progress(0.1, f"Reframe -> {nw}x{nh}...")
    _set_slicing(pipe, max(nw, nh))
    t0 = time.time()
    out = _qwen_call(pipe, prompt=prompt or "", image=canvas, mask_image=mask,
                     strength=float(strength), num_inference_steps=int(steps),
                     generator=_make_generator(seed), **_cfg(None)).images[0]
    if out.size != (nw, nh):
        out = out.resize((nw, nh), Image.LANCZOS)
    _log(f"reframe done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


@_gpu_serial
def outpaint(image, ratio_w, ratio_h, prompt, steps, seed):
    """Compat (CLI --reframe et appels existants): reframe en mode 'contain' (outpaint),
    borne a la resolution optimale du modele."""
    return reframe(image, ratio_w, ratio_h, "contain", prompt, steps, seed)


def outpaint_directions(image, mask, directions, prompt, steps, seed, strength=1.0, expand=0.3):
    """Outpaint directionnel (facon Fooocus): agrandit l'image dans les directions
    choisies parmi left/right/top/bottom, chacune de `expand` (fraction de la dimension
    d'origine), en repliquant les pixels du bord (mode 'edge'), puis fait remplir les
    bandes ajoutees par Z-Image (ZImageInpaintPipeline). Un `mask` peint (L, blanc = a
    changer) est optionnel: il est conserve dans la zone d'origine et combine avec les
    bandes ajoutees (blanches)."""
    img = np.array(image.convert("RGB"))
    H, W = img.shape[:2]
    m = np.array(mask.convert("L")) if mask is not None else np.zeros((H, W), dtype=np.uint8)
    dirs = set(d.lower() for d in (directions or []))
    if "top" in dirs:
        p = int(H * expand)
        img = np.pad(img, [[p, 0], [0, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[p, 0], [0, 0]], mode="constant", constant_values=255)
    if "bottom" in dirs:
        p = int(H * expand)
        img = np.pad(img, [[0, p], [0, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[0, p], [0, 0]], mode="constant", constant_values=255)
    if "left" in dirs:
        p = int(W * expand)
        img = np.pad(img, [[0, 0], [p, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[0, 0], [p, 0]], mode="constant", constant_values=255)
    if "right" in dirs:
        p = int(W * expand)
        img = np.pad(img, [[0, 0], [0, p], [0, 0]], mode="edge")
        m = np.pad(m, [[0, 0], [0, p]], mode="constant", constant_values=255)
    canvas = Image.fromarray(np.ascontiguousarray(img))
    mask_img = Image.fromarray(np.ascontiguousarray(m))
    full_size = canvas.size
    # Dilate un peu la zone a generer vers l'interieur -> le modele regenere une fine
    # bande de transition qui se raccorde a l'original (evite la jointure franche).
    from PIL import ImageFilter
    k = max(3, (int(min(full_size) * 0.02) // 2) * 2 + 1)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(min(k, 15)))
    # "Blurred edge fill": on remplit la zone a generer avec une version FLOUE de
    # l'extension du bord (memes couleurs/tonalite que l'original) au lieu d'un bord
    # replique net. Avec strength < 1.0 ce flou transparait -> continuite d'exposition
    # (plus de bande plus claire) et le modele ajoute le detail par-dessus.
    blur_r = max(8, int(min(full_size) * 0.03))
    canvas = Image.composite(canvas.filter(ImageFilter.GaussianBlur(blur_r)), canvas, mask_img)
    # Diffusion bornee a ~1 MP (zone optimale), puis recomposition: le centre (image
    # d'origine) garde sa pleine resolution, seuls les bords ajoutes sont generes.
    work_img, work_mask, _ = _cap_work_res(canvas, mask_img)
    w2, h2 = work_img.size
    pipe = get_pipe("inpaint")
    _log(f"outpaint {sorted(dirs)}: {image.size[0]}x{image.size[1]} -> "
         f"{full_size[0]}x{full_size[1]} (work {w2}x{h2}), {int(steps)} steps, "
         f"cfg {GUIDANCE:.1f} ...")
    _progress(0.1, f"Outpaint -> {full_size[0]}x{full_size[1]}...")
    _set_slicing(pipe, max(w2, h2))
    t0 = time.time()
    out = _qwen_call(pipe, prompt=prompt or "", image=work_img, mask_image=work_mask,
                     strength=float(strength), num_inference_steps=int(steps),
                     generator=_make_generator(seed), **_cfg(None)).images[0]
    out = _composite_back(out, canvas, mask_img, full_size,
                          feather=max(4, int(min(full_size) * 0.015)))
    _log(f"outpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def _make_generator(seed):
    return torch.Generator(DEVICE).manual_seed(int(seed)) if int(seed) >= 0 else None


@_gpu_serial
def _refine_whole(pipe, image, denoise, steps, prompt, seed):
    """Passe Qwen-Image img2img sur l'image entiere (ou une tuile). Le slicing est pose
    selon la taille reelle traitee: tuile 1024 -> OFF (rapide), whole 2K+ -> ON.
    IMPORTANT: on passe width/height = taille de l'image (alignee sur 16). Sinon Qwen-Image
    img2img retombe sur son defaut (height = default_sample_size * vae_scale_factor = 1024)
    et REDIMENSIONNE l'entree en 1024x1024 -> le ratio est ecrase (bug). En forcant les
    dimensions de l'entree, le ratio d'origine est preserve en upscale/img2img."""
    _set_slicing(pipe, max(image.size))
    w = round_to_multiple(image.width, 16)
    h = round_to_multiple(image.height, 16)
    return _qwen_call(
        pipe,
        prompt=prompt or "",
        image=image,
        width=w, height=h,
        strength=float(denoise),
        num_inference_steps=int(steps),
        generator=_make_generator(seed),
        **_cfg(None),
    ).images[0]


def _feather_mask_np(th, tw, overlap, left, right, top, bottom):
    """Masque (th, tw, 1) a rampe lineaire sur les bords qui jouxtent une autre tuile."""
    mask = np.ones((th, tw, 1), dtype=np.float32)
    f = int(overlap)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        if left:
            mask[:, :f, 0] *= ramp[np.newaxis, :]
        if right:
            mask[:, tw - f:, 0] *= ramp[::-1][np.newaxis, :]
        if top:
            mask[:f, :, 0] *= ramp[:, np.newaxis]
        if bottom:
            mask[th - f:, :, 0] *= ramp[::-1][:, np.newaxis]
    return mask


def _refine_tiled(pipe, image, denoise, steps, prompt, seed, tile, overlap):
    """Passe Z-Image en tuiles avec recomposition feather (facon Ultimate SD Upscale).
    Plafonne le pic VRAM (une tuile a la fois) et permet le 4K+ sans coutures.
    Memes rampe lineaire + overlap-add que esrgan_upscale, mais a scale 1 sur PIL."""
    w, h = image.size
    tile = round_to_multiple(tile)                       # multiple de 16 pour le VAE
    overlap = max(0, min(int(overlap), tile - 16))
    if w <= tile and h <= tile:
        # Une seule tuile = image entiere -> pas de duplication possible: denoise demande.
        return _refine_whole(pipe, image, denoise, steps, prompt, seed)
    # Anti-duplication 1: prompt vide par tuile (le prompt global decrit toute la compo).
    prompt = _tile_prompt(prompt)
    if not (prompt or "").strip():
        _log("refine tiled: prompt vide par tuile (anti-duplication; regle refine_tile_prompt).")
    # Anti-duplication 2 (filet): a fort denoise chaque tuile peut encore deriver.
    denoise = float(denoise)
    if _TILE_DENOISE_CAP > 0 and denoise > _TILE_DENOISE_CAP:
        _log(f"refine tiled: denoise {denoise:.2f} > plafond {_TILE_DENOISE_CAP:.2f} -> "
             f"reduit a {_TILE_DENOISE_CAP:.2f} (regle refine_tile_denoise_cap).")
        denoise = _TILE_DENOISE_CAP

    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    step = max(16, tile - overlap)
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    total = len(ys) * len(xs)
    _log(f"refine: tiled {w}x{h}, tile {tile} overlap {overlap} -> {len(xs)}x{len(ys)} = {total} tiles")
    i = 0
    for y in ys:
        for x in xs:
            if _STOP:
                _log("refine tiled: stop requested")
                break
            i += 1
            x2, y2 = min(x + tile, w), min(y + tile, h)
            x1, y1 = max(x2 - tile, 0), max(y2 - tile, 0)
            cw, ch = x2 - x1, y2 - y1
            _progress(0.45 + 0.5 * (i - 1) / max(1, total), f"Refine tile {i}/{total}")
            crop = image.crop((x1, y1, x2, y2))
            _t_tile = time.time()
            out = _refine_whole(pipe, crop, denoise, steps, prompt, seed)
            _log(f"  tile {i}/{total} ({cw}x{ch}) in {time.time() - _t_tile:.1f}s{_vram_str()}")
            if out.size != (cw, ch):
                out = out.resize((cw, ch), Image.LANCZOS)
            out_arr = np.asarray(out.convert("RGB"), dtype=np.float32) / 255.0
            mask = _feather_mask_np(ch, cw, overlap,
                                    left=x1 > 0, right=x2 < w, top=y1 > 0, bottom=y2 < h)
            acc[y1:y2, x1:x2, :] += out_arr * mask
            weight[y1:y2, x1:x2, :] += mask

    out = acc / np.clip(weight, 1e-6, None)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


# ----------------------------------------------------------------------------
# Orchestration : process_one, batch txt2img (run/_gen_meta restent dans app.py
# car run emet des gr.Error pour l'UI).
# ----------------------------------------------------------------------------
@_gpu_serial
def process_one(image, esrgan_model, factor, denoise, steps, prompt, seed, tile, overlap,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                do_esrgan=True, refine_first=False, apply_force_ratio=False):
    """Pipeline sur une PIL Image, renvoie (image, timings_dict).
    do_esrgan=False -> img2img pur (saute l'etage ESRGAN, refine sur l'image native).
    refine_first=True -> refine PUIS ESRGAN (la diffusion tourne a la resolution
    native = bien plus rapide), au lieu de ESRGAN PUIS refine (detail en haute-def).
    apply_force_ratio=True + FORCE_RATIO defini -> amene l'ENTREE au ratio choisi avant
    traitement: FORCE_RATIO_MODE 'crop' = recadrage centre (facon Fooocus), 'extend' =
    outpaint des bandes manquantes (rien n'est perdu). Sinon: ratio natif preserve."""
    timings = {"esrgan": 0.0, "refine": 0.0}
    image = image.convert("RGB")
    if apply_force_ratio and FORCE_RATIO:
        r = _parse_ratio(FORCE_RATIO)
        if r:
            _before = image.size
            if FORCE_RATIO_MODE == "extend":
                # max(6, steps): l'outpaint des bandes reste correct meme si l'upscale
                # tourne en pur ESRGAN (steps/denoise a ~0).
                image = _extend_to_ratio(image, r[0], r[1], prompt, max(6, int(steps)), seed)
                _verb = "extend (outpaint)"
            else:
                image = _crop_to_ratio(image, r[0], r[1])
                _verb = "crop"
            _log(f"force ratio {r[0]}:{r[1]} -> {_verb} {_before[0]}x{_before[1]} "
                 f"to {image.size[0]}x{image.size[1]}")
    w0, h0 = image.size
    use_esrgan = bool(do_esrgan and esrgan_model)
    do_refine = float(denoise) > 0.001
    _dbg(f"process_one in={w0}x{h0} factor={factor} denoise={denoise} steps={int(steps)} "
         f"do_esrgan={do_esrgan} refine_first={refine_first} esrgan={esrgan_model} "
         f"refine_tile={int(refine_tile)}")

    def _esrgan_stage(img):
        t0 = time.time()
        iw, ih = img.size
        _progress(0.15, f"ESRGAN upscale {iw}x{ih}...")
        model = load_esrgan(esrgan_model)
        _log(f"ESRGAN upscale: {iw}x{ih} (tile {int(tile)}) ...")
        up = esrgan_upscale(img, model, int(tile), int(overlap))
        # Cible = facteur applique a la taille d'origine (independant de l'ordre).
        target_w = round_to_multiple(w0 * factor)
        target_h = round_to_multiple(h0 * factor)
        up = up.resize((target_w, target_h), Image.LANCZOS)
        timings["esrgan"] += time.time() - t0
        _log(f"ESRGAN done in {timings['esrgan']:.1f}s -> {target_w}x{target_h}")
        return up

    def _refine_stage(img):
        t0 = time.time()
        pipe = load_pipe()
        rw, rh = img.size
        rt = int(refine_tile)
        # Garde-fou anti-crash: refine whole-image trop grand (4K+) -> auto-tuilage.
        if rt <= 0 and max(rw, rh) > _AUTO_TILE_ABOVE:
            rt = _pick_refine_tile(rw, rh, int(refine_overlap) or 64)
            _log(f"refine: image {rw}x{rh} > {_AUTO_TILE_ABOVE}px -> auto-tiling (tile {rt}) "
                 "pour eviter le pic VRAM (regles: auto_refine_tile_above, auto_refine_tile)")
        if rt > 0:
            out = _refine_tiled(pipe, img, denoise, steps, prompt, seed,
                                rt, int(refine_overlap) or 64)
        else:
            _log(f"Qwen refine: whole image {rw}x{rh}, denoise {float(denoise):.2f}, "
                 f"{int(steps)} steps ...")
            _progress(0.5, f"Qwen refine {rw}x{rh}...")
            out = _refine_whole(pipe, img, denoise, steps, prompt, seed)
        timings["refine"] += time.time() - t0
        return out

    result = image
    if refine_first:
        # refine sur l'image native (rapide) puis agrandissement ESRGAN.
        if do_refine:
            result = _refine_stage(result)
        if use_esrgan:
            result = _esrgan_stage(result)
    else:
        # ordre classique: ESRGAN (detailleur) puis refine a la resolution agrandie.
        if use_esrgan:
            result = _esrgan_stage(result)
        if do_refine:
            result = _refine_stage(result)

    if not use_esrgan and not do_refine:
        _log(f"process_one: nothing to do (no ESRGAN, denoise=0) on {w0}x{h0}")

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _progress(1.0, "Done")
    _log(f"process_one done | esrgan {timings['esrgan']:.1f}s + refine {timings['refine']:.1f}s "
         f"= {timings['esrgan'] + timings['refine']:.1f}s")
    return result, timings


@_gpu_serial
def txt2img_run(prompt, width, height, gen_steps, seed, negative_prompt="",
                upscale=False, esrgan_model=None, factor=2.0, denoise=0.30, steps=12,
                tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                refine_first=False):
    """Genere une image (txt2img Z-Image) puis, si upscale=True, la passe dans le
    pipeline ESRGAN + refine. Renvoie (image, timings_dict)."""
    timings = {"txt2img": 0.0, "esrgan": 0.0, "refine": 0.0}
    t0 = time.time()
    base = generate(prompt, width, height, gen_steps, seed, negative_prompt)
    timings["txt2img"] = time.time() - t0
    if not upscale:
        return base, timings
    result, t = process_one(base, esrgan_model, factor, denoise, steps, prompt, seed,
                            tile, overlap, refine_tile=refine_tile, refine_overlap=refine_overlap,
                            refine_first=refine_first)
    timings["esrgan"] = t.get("esrgan", 0.0)
    timings["refine"] = t.get("refine", 0.0)
    return result, timings


def _gen_meta(mode, prompt, negative="", seed=None, steps=None, guidance=None,
              size=None, model=None, styles=None, extra=None):
    """Construit le dict de metadonnees de generation (pour sidecar/PNG)."""
    m = {"app": "crispz-klein", "mode": mode, "prompt": prompt or "",
         "negative": negative or "", "date": _now_stamp()}
    if seed is not None and int(seed) >= 0:
        m["seed"] = int(seed)
    if steps is not None:
        m["steps"] = int(steps)
    if guidance is not None:
        m["guidance"] = float(guidance)
    if size:
        m["size"] = f"{size[0]}x{size[1]}"
    # Noms de styles appliques (en plus des mots-cles deja injectes dans le prompt).
    _styles = [s for s in (styles or []) if s and s not in ("None", "none")]
    if _styles:
        m["styles"] = _styles
    m["sampler"] = f"{SAMPLER}/{SCHEDULE}"
    m["model"] = model or (ZIMAGE_TRANSFORMER or BASE_REPO)
    if LORAS:
        m["loras"] = [f"{os.path.basename(p)}@{w}" for p, w in LORAS]
    if extra:
        m.update(extra)
    return m
