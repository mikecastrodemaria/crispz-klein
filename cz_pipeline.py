"""crispz-klein - the FLUX.2 Klein core (diffusers, BF16): loading the pipelines
(txt2img / img2img / inpaint) + multi-reference editing (Omni/Edit tab) + LoRAs /
checkpoints / transformer, generation and orchestration (generate / txt2img_run /
process_one / outpaint / inpaint) + the mutable runtime state.

Fork of crispz-qwen-edit (Qwen-Image). Mapping:
  - base txt2img             -> Flux2KleinPipeline
  - Omni/Edit tab            -> Flux2KleinPipeline  (the SAME object: `image` takes a
                                LIST of PIL images -> multi-reference is native, no
                                second model)
  - inpaint / reframe        -> Flux2KleinInpaintPipeline
  - img2img (refine/upscale) -> Flux2KleinInpaintPipeline + a full WHITE mask
                                (the base pipeline does NOT expose `strength`)

klein-4B is DISTILLED (`is_distilled: true`). Measured on 2026-09-05 (tests/
test_klein_guidance.py, RTX 5090): guidance_scale 1.0 / 4.0 / 8.0 -> bit-for-bit
identical images (MAE 0.0000), with diffusers itself saying "Guidance scale is
ignored for step-wise distilled models". So there is NEITHER a usable CFG NOR a
usable negative prompt: `_cfg` returns {} and the protocol announces
supports.negative = False. The UI's "guidance" slider is kept (cz_ui's API
contract) but has no effect on the render.

The module's public API stays identical to upstream (same names, e.g.
ZIMAGE_TRANSFORMER, generate_omni, OMNI_MODEL, SAMPLER_CHOICES) so that neither
cz_ui nor cz_cli nor cz_protocol breaks. The "omni" symbols survive but now point
at the SAME model as the base.

app reads the current state through cz_pipeline.NAME (BASE_REPO, ZIMAGE_TRANSFORMER,
...) and sets cz_pipeline._PROGRESS / cz_pipeline._STOP from the UI handlers.
Depends only on cz_core / cz_esrgan / cz_imageio (never on app or gradio).

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
    _prefs, _is_single_file, _looks_single_file, _log, _dbg,
)

# The base FLUX.2 Klein model (txt2img/img2img/inpaint/edit). Overridden through env
# ZIMAGE_MODEL (compat) or KLEIN_MODEL, or prefs. Public repo, Apache 2.0.
# CAREFUL: do NOT switch to FLUX.2-klein-9B (non-commercial licence), see FORK.md.
DEFAULT_BASE_REPO = (os.environ.get("KLEIN_MODEL") or "black-forest-labs/FLUX.2-klein-4B")
# Multi-reference editing has NO separate model on klein: the base pipeline's `image`
# takes a list of PIL images. The "omni" default is therefore the base model itself
# (the symbol survives for cz_ui / cz_protocol, see the module docstring).
DEFAULT_OMNI_REPO = DEFAULT_BASE_REPO
from cz_esrgan import load_esrgan, esrgan_upscale
from cz_imageio import _now_stamp
import cz_hw

# Speed: allow TF32 (matmul/cudnn) on the GPU. A free win on Ampere+ for the residual
# fp32 operations; the weights stay BF16. No effect outside CUDA.
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# The current FLUX.2 Klein model. An HF repo / diffusers folder -> BASE_REPO. A
# single-file checkpoint (a Civitai .safetensors) passed as the "model" -> a transformer
# override (the VAE and the Qwen3 encoder still come from the base repo).
# Config/env keys. The 'zimage_*' names are leftovers from crispz-studio (Z-Image): in a
# FLUX.2 fork they mean nothing any more, and an error message saying "set
# 'zimage_model'" is incomprehensible. The proper names are 'klein_model' /
# 'klein_transformer' (env KLEIN_MODEL / KLEIN_TRANSFORMER); the old ones are still READ
# so no existing config breaks, and they are reported when they answer.
# NB: the Python VARIABLES keep their names (ZIMAGE_TRANSFORMER...) - cz_ui, cz_cli and
# cz_protocol import them, that is the module's API contract (see the docstring).
CFG_MODEL_KEY = "klein_model"
CFG_TRANSFORMER_KEY = "klein_transformer"


def _cfg_first(*keys, env=()):
    """First non-empty value among the env variables, then the prefs/config keys.
    Logs when it is an old name that answered."""
    for e in env:
        v = (os.environ.get(e) or "").strip()
        if v:
            return v
    for i, k in enumerate(keys):
        v = (_prefs.get(k) or CONFIG.get(k) or "")
        v = v.strip() if isinstance(v, str) else v
        if v:
            if i:
                _log(f"config: '{k}' is the old crispz-studio name, rename it to "
                     f"'{keys[0]}' (still read for now)")
            return v
    return None


_zmodel = _cfg_first(CFG_MODEL_KEY, "zimage_model",
                     env=("KLEIN_MODEL", "ZIMAGE_MODEL")) or DEFAULT_BASE_REPO
ZIMAGE_TRANSFORMER = _cfg_first(CFG_TRANSFORMER_KEY, "zimage_transformer",
                                env=("KLEIN_TRANSFORMER", "ZIMAGE_TRANSFORMER"))
if _is_single_file(_zmodel):
    ZIMAGE_TRANSFORMER = _zmodel
    BASE_REPO = DEFAULT_BASE_REPO
else:
    BASE_REPO = _zmodel

# Replacement text encoder (Models > Checkpoints > Text encoder). Empty = the one from
# the base repo, as before. Otherwise a FOLDER in transformers format (config.json +
# weights) or an HF repo ('owner/repo', 'owner/repo/subfolder') -- e.g. an "abliterated"
# Qwen3 of the same size. Only the encoder changes: tokenizer, VAE and transformer stay
# those of the base repo.
CFG_TEXT_ENCODER_KEY = "text_encoder"


def _resolve_text_encoder(env, prefs, config):
    """The encoder at startup: env > preferences > config. A key PRESENT in the
    preferences wins even when empty: that is the "Default" choice made in the UI, and a
    config.txt value must not undo it on the next start (a "" used to pass for absent)."""
    v = str(env.get("KLEIN_TEXT_ENCODER") or "").strip()
    if v:
        return v
    if CFG_TEXT_ENCODER_KEY in prefs:
        return str(prefs.get(CFG_TEXT_ENCODER_KEY) or "").strip()
    return str(config.get(CFG_TEXT_ENCODER_KEY) or "").strip()


TEXT_ENCODER = _resolve_text_encoder(os.environ, _prefs, CONFIG)
# The one REALLY loaded ('' = the base repo's). Distinct from TEXT_ENCODER: an encoder
# that does not suit the current repo is dropped at load time, and the metadata says what
# ran, not what was asked for.
_TEXT_ENCODER_ACTIVE = ""
TEXT_ENCODERS_DIR = (os.environ.get("TEXT_ENCODERS_DIR") or _prefs.get("text_encoders_dir")
                     or CONFIG.get("text_encoders_dir") or "").strip()

# Model folders: single-file checkpoints to switch between + LoRAs to apply.
CHECKPOINTS_DIR = (os.environ.get("CHECKPOINTS_DIR") or _prefs.get("checkpoints_dir")
                   or CONFIG.get("checkpoints_dir") or os.path.join(HERE, "checkpoints"))
# Additional checkpoints folder (optional) -> merged with CHECKPOINTS_DIR into the same
# checkpoint list. Empty by default; configurable through UI / prefs / config / env.
CHECKPOINTS_EXTRA_DIR = (os.environ.get("CHECKPOINTS_EXTRA_DIR") or _prefs.get("checkpoints_extra_dir")
                         or CONFIG.get("checkpoints_extra_dir") or "").strip()
LORAS_DIR = (os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
             or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))


def _split_dirs(spec):
    """Folder list from a JSON list or from an 'a;b' string (os.pathsep or ';')."""
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


# EXTRA LoRA folders (e.g. the Civitai library shared with other tools): env
# LORAS_EXTRA_DIRS ('a;b') > preferences > config 'loras_extra_dirs'. Merged with
# LORAS_DIR into a single list; on a duplicate name, LORAS_DIR wins.
LORAS_EXTRA_DIRS = _split_dirs(os.environ["LORAS_EXTRA_DIRS"] if "LORAS_EXTRA_DIRS" in os.environ
                               else (_prefs.get("loras_extra_dirs")
                                     or CONFIG.get("loras_extra_dirs")))


def _lora_dirs():
    """LoRA folders to scan: the main one + the extras, deduplicated, in priority order."""
    dirs = [LORAS_DIR]
    for d in LORAS_EXTRA_DIRS:
        if d and d not in dirs:
            dirs.append(d)
    return dirs


def resolve_lora_path(name):
    """Path of a LoRA from a slot name: an absolute path as is, otherwise the relative
    name (subfolders included) looked up in LORAS_DIR then in the extras. Absent
    everywhere, the path inside LORAS_DIR (the caller reports 'not found')."""
    name = str(name or "")
    if os.path.isabs(name):
        return name
    for d in _lora_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(LORAS_DIR, name)


# Active LoRAs: a list of (path, weight). Several LoRAs can be combined (multi-slot).
LORAS = []
LORA_WEIGHT = float(CONFIG.get("default_lora_weight", 1.0))  # the slots' default weight


def _lora_weight_range():
    """Bounds of the LoRA weight sliders (config 'lora_weight_min'/'lora_weight_max').
    Default -2..2: NEGATIVE weights are valid and useful (they invert the LoRA's
    effect). Defensive: unreadable values or min >= max -> fall back to the default."""
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
# The default weight has to stay inside the bounds (or the slider would be born out of range).
LORA_WEIGHT = min(LORA_WEIGHT_MAX, max(LORA_WEIGHT_MIN, LORA_WEIGHT))
# LoRAs applied AT STARTUP (e.g. Lightning 8-step). config 'default_loras' = a list of
# names (inside LORAS_DIR) or of [name, weight] pairs. Resolved to (path, weight).
for _spec in (CONFIG.get("default_loras") or []):
    _nm, _w = (_spec if isinstance(_spec, (list, tuple)) and len(_spec) == 2
               else (_spec, LORA_WEIGHT))
    if _nm and _nm not in ("None", "none"):
        _p = resolve_lora_path(_nm)
        if os.path.isfile(_p):
            LORAS.append((_p, float(_w)))
# The Omni/Edit model. On klein there is NO second model: multi-reference editing is
# served by the base pipeline. OMNI_MODEL therefore follows BASE_REPO and exists mostly
# so that cz_ui / cz_protocol keep their contract (never empty -> edit available).
OMNI_MODEL = BASE_REPO

# Process-wide caches. A "base" pipeline (txt2img Flux2KleinPipeline) owns the
# components; img2img / inpaint derive from it through from_pipe -> shared weights, no
# duplicate VRAM. Cache key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE, LORAS).
_BASE_PIPE = None
_DERIVED = {}
_LOADED_KEY = None
# LoRAs actually applied on _BASE_PIPE (a list of (path, weight)). Used to hot-swap the
# LoRAs without reloading the model: if it diverges from LORAS, _apply_loras resyncs.
_APPLIED_LORAS = []
# LoKrs MERGED into the current transformer (a list of (path, weight)). This is NOT an
# adapter: once added to the weights it cannot be removed. _apply_loras therefore compares
# this set with the requested one and forces a full reload as soon as it changes.
_APPLIED_LOKRS = []
# EDIT LoRAs: a set SEPARATE from the base one. On klein editing goes through the SAME
# pipeline (native multi-reference), but the edit LoRAs stay a distinct set, applied and
# removed around an edit call without touching the generation LoRAs. Same (path, weight)
# format. EDIT_LORAS_ENABLED = the UI's "Edit LoRAs" checkbox: OFF -> the set is
# remembered but not applied (so with/without can be compared in one click).
EDIT_LORAS = []
EDIT_LORAS_ENABLED = bool(CONFIG.get("edit_loras_enabled", True))
_APPLIED_EDIT_LORAS = []
# FAST edit mode (the 'Edit speed' dropdown): None = off (steps/guidance from
# Settings), otherwise {"name", "steps", "guidance", "path"} - path = the Lightning LoRA
# to stack on the edit pipe (None for 'Auto': an already distilled model, Rapid-AIO or a
# Lightning merge, whose model_profiles profile sets steps/guidance).
EDIT_SPEED = None

# Step 2 (VRAM coexistence): CPU offload of the diffusion pass. none = everything in
# VRAM. model = unloads per submodule (a good compromise). sequential = more aggressive,
# slower. This is NOT quantization: the weights stay BF16, they travel RAM <-> GPU.
# klein-4B fits in VRAM (~15 GB) but the 9B does not (~35 GB). 'auto' (the default) =
# a FREE VRAM test at load time (cz_hw, based on _base_vram_need_gb): a model that
# overflows does not crash under Windows, it spills into shared RAM (Sysmem Fallback) and
# renders 50-100x slower WITHOUT a message -> 'none' is only promoted once the card has
# proven it has the room. _effective_offload also corrects outright when the base repo
# does not fit in TOTAL VRAM. Resolution order (the first one set wins):
# explicit UI/CLI choice > env CZ_OFFLOAD > config default_cpu_offload > auto.
OFFLOAD_CHOICES = ("auto", "none", "model", "sequential")
OFFLOAD_MODE = ((os.environ.get("CZ_OFFLOAD") or "").strip()
                or str(CONFIG.get("default_cpu_offload", "") or "").strip()).lower() or "auto"
if OFFLOAD_MODE not in OFFLOAD_CHOICES:
    _log(f"CZ_OFFLOAD/default_cpu_offload '{OFFLOAD_MODE}' unknown -> auto")
    OFFLOAD_MODE = "auto"
# The concrete mode resolved for 'auto' (set by _resolve_auto on the first load) and the
# flag of the runtime safety net (set by the VRAM callback during the denoise).
_AUTO_OFFLOAD = ""
_VRAM_DOWNGRADE = False

# Guidance. klein-4B is DISTILLED: diffusers IGNORES guidance_scale (checked, see the
# module docstring + tests/test_klein_guidance.py -> bit-for-bit identical images from 1.0
# to 8.0). The variable is kept because cz_ui / cz_cli / cz_protocol read and display it,
# but _cfg() does NOT pass it to the pipeline any more: it has no effect.
# Overridable through env KLEIN_CFG (no effect either, kept for symmetry).
GUIDANCE = float(os.environ.get("KLEIN_CFG") or CONFIG.get("default_guidance") or 0) or 1.0

# Forced ratio (Fooocus-style) for upscale/img2img: when set, the INPUT image is
# center-cropped to that ratio before processing (crop to fit). Empty = the native ratio is
# preserved (the default). Format: 'W:H' or 'WxH' (e.g. '13:19', '832x1216'). Driven by the
# UI (checkbox + Aspect ratio dropdown) through set_force_ratio, or by config.txt
# 'force_upscale_ratio'.
FORCE_RATIO = (os.environ.get("CZ_FORCE_RATIO") or CONFIG.get("force_upscale_ratio") or "").strip()
# How to reach the forced ratio: 'crop' = center crop (loses the edges, the default),
# 'extend' = extends the image to the ratio by outpainting (loses nothing, adds generated
# bands). UI (radio) through set_force_ratio_mode, config 'force_ratio_mode'.
FORCE_RATIO_MODE = (os.environ.get("CZ_FORCE_RATIO_MODE")
                    or CONFIG.get("force_ratio_mode") or "crop").strip().lower()
# Seam-blending pass of the extend mode: after the bands are outpainted, a LIGHT img2img
# pass runs on the extended image and ONLY the bands plus a feathered transition margin are
# pasted back from it (the original centre stays untouched). 0 = off.
try:
    EXTEND_DENOISE = float(CONFIG.get("force_ratio_extend_denoise", 0.22) or 0.0)
except Exception:
    EXTEND_DENOISE = 0.22

# Sampler / scheduler. The FLUX.2 pipeline imposes a custom `sigmas` schedule: only the
# schedulers whose set_timesteps accepts `sigmas` work. In practice -> Euler flow-matching
# (native, the default), UniPC (multistep) and LCM flow-matching (interesting on
# distilled/Turbo models: few steps, guidance ~0-1).
# diffusers' DPM++ 2M / DPM2a / DPM++ SDE (dpmpp_sde) do NOT take custom sigmas ->
# incompatible (DPMSolverSDEScheduler also requires torchsde). Not exposed.
SAMPLER_CHOICES = ("euler", "unipc", "lcm")
SAMPLER = (os.environ.get("ZIMAGE_SAMPLER") or CONFIG.get("default_sampler") or "euler").strip().lower()
if SAMPLER not in SAMPLER_CHOICES:
    SAMPLER = "euler"

# Sigma schedule (= the "scheduler" in ComfyUI terms). sgm_uniform = FLUX.2's native one
# (linspace + dynamic shift). beta/karras/exponential = a sigma remapping applied ON TOP of
# the pipeline's schedule (FlowMatchEuler/UniPC: use_*_sigmas). beta -> scipy.
SCHEDULE_CHOICES = ("sgm_uniform", "beta", "karras", "exponential")
# 'simple' (ComfyUI) names EXACTLY the native schedule exposed here as 'sgm_uniform': the
# default sigmas the pipeline hands the scheduler are linspace(1, 1/n, n), which is what
# ComfyUI calls 'simple' on a flow-matching model. Accepted as input everywhere
# (config/env/CLI/XYZ) so a CivitAI recipe can be copied word for word, but normalised to
# the canonical name: metadata and presets only ever carry one name.
_SCHEDULE_ALIASES = {"simple": "sgm_uniform"}
SCHEDULE_INPUTS = SCHEDULE_CHOICES + tuple(_SCHEDULE_ALIASES)   # listes ouvertes (CLI/XYZ)


def _norm_schedule(name, default="sgm_uniform"):
    """Nom de schedule -> nom canonique (alias resolus). Inconnu -> `default`."""
    n = (name or "").strip().lower()
    n = _SCHEDULE_ALIASES.get(n, n)
    return n if n in SCHEDULE_CHOICES else default


SCHEDULE = _norm_schedule(os.environ.get("ZIMAGE_SCHEDULE") or CONFIG.get("default_schedule"))
_SCHEDULE_FLAG = {"beta": "use_beta_sigmas", "karras": "use_karras_sigmas",
                  "exponential": "use_exponential_sigmas"}  # sgm_uniform -> no flag (native)
# The model's own scheduler config (captured on the first load) -> the base every other
# sampler is built from (keeps shift/flow params whatever the current sampler is).
_BASE_SCHED_CONFIG = None

# UI progress hook (gradio gr.Progress). None outside the UI (CLI/server). Set by
# the handlers through cz_pipeline._PROGRESS = ...
_PROGRESS = None
# Fooocus-style Stop: a global flag plus the interruption of the diffusers pipelines. Set
# by the handlers through cz_pipeline._STOP = ... and by request_stop().
_STOP = False

# GPU lock: serialises EVERY generation. Gradio does not serialise the events of
# different LISTENERS (manual Generate vs Run queue vs the detailer): two threads can then
# call the SAME shared pipeline and step the SAME scheduler -> its index runs past the end
# ("IndexError: index 31 is out of bounds for dimension 0 with size 31",
# scheduling_flow_match_euler_discrete.step). RLock: one thread's nested calls
# (txt2img_run -> generate, process_one -> _refine_whole) stay free.
_GPU_LOCK = threading.RLock()


def _gpu_serial(fn):
    """Decorator: runs fn under _GPU_LOCK (a single GPU generation at a time)."""
    import functools

    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        with _GPU_LOCK:
            return fn(*args, **kwargs)
    return _locked


def _gpu_exclusive(fn):
    """Decorator for the setters that RELEASE the shared pipeline (free_vram and the three
    that call it). Doing that under a running denoise loop pulls the weights out from under
    it; the scheduler race showed what touching the shared pipe mid-render costs.

    Unlike the sampler, these cannot be DEFERRED: you pressed Free VRAM, or changed the
    encoder, to have it happen -- so they WAIT. The wait is announced, because a handler
    blocked for a whole render looks frozen otherwise. The try-acquire first keeps the
    common case silent.

    RLock -> a call from INSIDE a generation goes straight through: retry_on_oom and
    _consume_vram_downgrade both free the VRAM on the generation's own thread, and that
    thread is between two pipeline calls, not inside one.

    NOT applied to set_loras() nor set_zimage_transformer(): checked, they only write a
    global that the next _ensure_base reads under the lock, so a running render is not
    affected. A PAIR of setters is still two operations, though -- nothing makes
    set_zimage_transformer('') + set_zimage_model(x) atomic together.
    """
    import functools

    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        if not _GPU_LOCK.acquire(blocking=False):
            _log(f"{fn.__name__}: waiting for the render in progress (releasing the "
                 f"shared pipeline now would break it) ...")
            _GPU_LOCK.acquire()
        try:
            return fn(*args, **kwargs)
        finally:
            _GPU_LOCK.release()
    return _locked

# Seed handling (Fooocus-style):
#  _LAST_SEED         = the CONCRETE seed of the last render (a random -1 is resolved to a
#                       real value) -> the "Reuse last seed" button + honest metadata.
#  _NO_SEED_INCREMENT = True -> a whole batch uses the same seed (no +i per image).
_LAST_SEED = -1
_NO_SEED_INCREMENT = False
# True -> in txt2img+upscale, ALSO save the original txt2img image (before the upscale).
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
    """CFG kwargs. On klein: NONE.

    klein-4B is step-wise distilled -> diffusers ignores `guidance_scale` (measured:
    1.0 / 4.0 / 8.0 give bit-for-bit identical images, see
    tests/test_klein_guidance.py) and the Flux2Klein* pipelines do NOT expose
    `negative_prompt` (only `negative_prompt_embeds`, useless without CFG).

    So {} is returned rather than passing kwargs with no effect or making diffusers
    warn on every call. The signature is kept: every upstream callsite
    (`**_cfg(negative)`) stays valid, and a negative that does come in is ignored
    SILENTLY here but ANNOUNCED upstream by cz_protocol (supports.negative = False +
    a warning on a spec that carries one) -- house rule: degradation announced,
    never silent.
"""
    if negative:
        _dbg("negative prompt ignored: klein is distilled (no CFG, no negative_prompt)")
    return {}


# --- Prompt embedding cache -------------------------------------------------------
# Encoding a prompt sends the text encoder through the GPU. Under 'model' offload that
# transfer is paid on EVERY pipeline call -- including the detailer's passes, which
# redo the SAME prompt once per face and per hand.
# Measured on crispz-klein (9B GGUF, model offload): prompt+setup 5.2-6.1 s per pass
# without the cache against 1.7-1.8 s with it, for 0.3 s of diffusion. 2.1x per hand.
# Without offload the win drops to ~8 % (the encoder is already resident, nothing to
# move).
#
# encode_prompt() short-circuits the encoder as soon as it is handed its embeddings. So
# the TUPLE it returns is memorised and handed back to __call__ through _EMBED_OUTS.
# The tensors are kept in RAM (a few MB): they hold no VRAM and survive the offload's
# moves.
# Flux2Klein: encode_prompt -> (prompt_embeds, text_ids); text_ids is recomputed from
# the embeddings, no need to keep it.
_EMBED_OUTS = ("prompt_embeds",)
_CFG_IGNORED_SAID = set()   # guidance values already reported as inert
_CFG_REAL_SAID = set()      # (checkpoint, guidance) already announced as real CFG
_EMBED_CACHE = {}
_EMBED_CACHE_MAX = max(0, int(CONFIG.get("prompt_embed_cache", 8) or 0))


def _embed_cache_clear(why=""):
    """Empties the cache. Called as soon as the encoder may have changed (base repo, VRAM
    release): an embedding computed by another encoder is wrong."""
    if _EMBED_CACHE:
        _dbg(f"prompt embed cache cleared ({len(_EMBED_CACHE)} entries){why}")
    _EMBED_CACHE.clear()


def _cached_prompt_embeds(pipe, prompt, kw):
    """Embeddings of `prompt` for this pipeline, computed once then reused.

    Returns a kwargs dict for __call__, or None when the cache is off, when the
    pipeline does not expose the expected API, or when the encoding fails: in all
    those cases the caller passes the prompt as text and nothing changes. A cache must
    never break a render.
"""
    if not _EMBED_CACHE_MAX or not _EMBED_OUTS:
        return None
    try:
        enc = getattr(pipe, "text_encoder", None)
        if enc is None or not hasattr(pipe, "encode_prompt"):
            return None
        # The LoRAs are part of the key: some of them touch the text encoder, and an
        # embedding computed without them would be wrong.
        # So is the replacement encoder: id(enc) alone is not enough, CPython recycles
        # the id of a freed object -- and another encoder encodes differently.
        key = (BASE_REPO, _TEXT_ENCODER_ACTIVE, id(enc), prompt,
               kw.get("max_sequence_length"),
               tuple(sorted((p, float(w)) for p, w in _APPLIED_LORAS)))
        hit = _EMBED_CACHE.get(key)
        if hit is None:
            out = pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)
            if not isinstance(out, (tuple, list)):
                out = (out,)
            hit = tuple(v.detach().to("cpu") if hasattr(v, "detach") else v
                        for v in out[:len(_EMBED_OUTS)])
            if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
                _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))      # FIFO, borne simple
            _EMBED_CACHE[key] = hit
            _dbg(f"prompt embeds computed and cached ({len(_EMBED_CACHE)}/"
                 f"{_EMBED_CACHE_MAX}) for {prompt[:40]!r}")
        else:
            _dbg(f"prompt embeds reused (text encoder not touched) for {prompt[:40]!r}")
        dev = pipe._execution_device
        return {name: (v.to(dev) if hasattr(v, "to") else v)
                for name, v in zip(_EMBED_OUTS, hit) if name}
    except Exception as e:
        _dbg(f"prompt embed cache off for this call ({type(e).__name__}: {e})")
        return None


def _qwen_call(pipe, **kw):
    """Calls a Flux2Klein pipeline, absorbing the two API gaps with upstream.

    1. Implicit white mask: `Flux2KleinPipeline` does NOT expose `strength`, so
       img2img goes through `Flux2KleinInpaintPipeline`. A call carrying `image` +
       `strength` WITHOUT `mask_image` is an img2img -> a fully WHITE mask
       (everything is redrawn) the size of the image is injected.
    2. Leftover CFG kwargs: if a callsite (or an upstream merge) hands over
       `true_cfg_scale` / `negative_prompt`, they are dropped and the call is retried
       instead of crashing the generation.

    The name `_qwen_call` is kept to limit the conflict surface when merging from
    `qwen/main` (some thirty callsites).
"""
    # guidance_scale: klein is distilled -> diffusers IGNORES any value > 1.0 and logs a
    # warning on EVERY call. So 1.0 is passed explicitly (= no CFG, the model's truth) to
    # silence the noise. Should an UNdistilled FLUX.2 checkpoint ever be loaded,
    # is_distilled is false and the UI slider is passed through, becoming a real CFG again.
    need_cfg = False
    if "guidance_scale" not in kw:
        try:
            distilled = bool(getattr(pipe.config, "is_distilled", False))
        except Exception:
            distilled = True
        want = float(GUIDANCE)
        # `is_distilled` describes the BASE REPO, not the loaded transformer. With a
        # single-file override it says nothing about the model doing the computing any
        # more: community checkpoints are explicitly NOT distilled ("undistilled, use
        # with Turbo LoRA") and need a real CFG plus many more steps. Forcing them to 1.0
        # rendered a blurry mush, without a word. If the user raised the guidance AND
        # loaded an override, it is passed through: on the base repo we know it is inert
        # (measured bit-for-bit), on their checkpoint we do not.
        if distilled and want > 1.0 and ZIMAGE_TRANSFORMER:
            # Passing guidance_scale was NOT ENOUGH. The pipeline decides by itself:
            #   do_classifier_free_guidance = guidance > 1 and not config.is_distilled
            # and config.is_distilled is the BASE REPO's (True for klein). The pass
            # without the prompt was therefore never done. The log said "passed through",
            # diffusers answered on the next line "ignored for step-wise distilled
            # models" -- caught on the 2026-09-10 bench, where kleinForeskin at 28 steps
            # cost 0.6 s/step like a distilled one instead of double. The flag is raised
            # for the duration of the call (see _run below).
            need_cfg = True
            said = (str(ZIMAGE_TRANSFORMER), want)
            if said not in _CFG_REAL_SAID:
                _CFG_REAL_SAID.add(said)
                _log(f"guidance {want:g} applied as REAL CFG on "
                     f"{os.path.basename(str(ZIMAGE_TRANSFORMER))}: two passes per "
                     f"step (with and without the prompt), so ~2x the diffusion time. "
                     f"That is how an 'undistilled' checkpoint runs. On a DISTILLED "
                     f"checkpoint, a guidance > 1 degrades the image: set it back to 1.0.")
            kw["guidance_scale"] = want
        else:
            if distilled and want > 1.0 and want not in _CFG_IGNORED_SAID:
                _CFG_IGNORED_SAID.add(want)
                _log(f"guidance {want:g} ignored: {BASE_REPO} is distilled and CFG is inert "
                     f"there (measured bit-for-bit identical from 1.0 to 8.0). It does "
                     f"apply on a single-file checkpoint, though.")
            kw["guidance_scale"] = 1.0 if distilled else want
    if "strength" in kw and kw.get("image") is not None and "mask_image" not in kw:
        img = kw["image"]
        ref = img[0] if isinstance(img, (list, tuple)) else img
        kw["mask_image"] = Image.new("L", ref.size, 255)
        _dbg(f"img2img -> inpaint pipeline + a full white mask {ref.size}")
    # Reuse the embeddings if this prompt has already been encoded (see _EMBED_CACHE).
    # Passing them skips the text encoder: that is the whole win.
    if isinstance(kw.get("prompt"), str) and not any(k in kw for k in _EMBED_OUTS):
        _emb = _cached_prompt_embeds(pipe, kw["prompt"], kw)
        if _emb:
            kw.update(_emb)
            kw["prompt"] = None
    # Real CFG: without negative embeddings handed in, the pipeline encodes an empty
    # negative ITSELF on every call (it imposes it, there is no negative_prompt input) --
    # under offload the encoder would climb back onto the GPU for every image and cancel
    # the cache. Same cache as the positive, with the empty string as the key.
    if need_cfg and "negative_prompt_embeds" not in kw:
        _neg = _cached_prompt_embeds(pipe, "", kw)
        if _neg and "prompt_embeds" in _neg:
            kw["negative_prompt_embeds"] = _neg["prompt_embeds"]

    # Safety net: a pipe left on the CPU by an earlier failure (LoRA, offload) would make
    # THIS call fail on "Cannot generate a cpu tensor from a generator of type cuda".
    restore_offload(pipe, "an earlier failure")

    def _run():
        # Encode / diffusion / decode breakdown, in debug only. A total ("50s") does not
        # say what to optimise: on an offloaded base, moving the Qwen3 encoder then the
        # transformer costs a FIXED time that neither the steps nor the resolution reduce.
        # Knowing where the time goes is knowing whether lowering the steps helps at all.
        if cz_core.LOG_LEVEL >= 2 and "callback_on_step_end" not in kw:
            marks = {"t0": time.time()}

            def _mark(_pipe, i, _t, kwargs):
                marks.setdefault("first_step", time.time())
                marks["last_step"] = time.time()
                return kwargs
            kw["callback_on_step_end"] = _mark
            try:
                out = pipe(**kw)
            except TypeError as e:
                if "callback_on_step_end" not in str(e):
                    raise
                kw.pop("callback_on_step_end", None)   # a pipeline with no callback -> never mind
                marks.clear()
                out = pipe(**kw)
            if marks.get("first_step"):
                _dbg(f"phases: prompt+setup {marks['first_step'] - marks['t0']:.1f}s | "
                     f"diffusion {marks['last_step'] - marks['first_step']:.1f}s | "
                     f"decode {time.time() - marks['last_step']:.1f}s")
            return out
        try:
            return pipe(**kw)
        except TypeError as e:
            # callback_on_step_end = the optional VRAM guard (see _vram_guard_kwargs):
            # an old diffusers build that does not know it runs without the guard.
            if any(k in kw for k in ("true_cfg_scale", "negative_prompt", "callback_on_step_end")):
                for k in ("true_cfg_scale", "negative_prompt", "callback_on_step_end"):
                    kw.pop(k, None)
                _dbg(f"klein call: retrying without the optional kwargs ({e})")
                return pipe(**kw)
            raise

    if not need_cfg:
        return _run()
    reg = getattr(pipe, "register_to_config", None)
    if reg is None:
        _log("real CFG impossible: this pipeline has no register_to_config -- "
             "diffusers will ignore the guidance")
        return _run()
    was = bool(getattr(pipe.config, "is_distilled", True))
    reg(is_distilled=False)
    try:
        return _run()
    finally:
        # The pipeline is shared (a process-wide cache): a flag left raised would put
        # every later call in CFG, including on the base repo.
        reg(is_distilled=was)


def _scheduler_accepts_sigmas(sched):
    """The FLUX.2 pipeline calls set_timesteps(..., sigmas=<custom schedule>). A scheduler
    whose set_timesteps does not accept `sigmas` crashes at generation time."""
    import inspect
    try:
        return "sigmas" in inspect.signature(sched.set_timesteps).parameters
    except Exception:
        return False


def _build_scheduler(sampler, schedule, config):
    """Builds the chosen scheduler (sampler x schedule) from the model's native config.
    schedule (sgm_uniform/beta/karras/exponential) = a sigma remapping (use_*_sigmas)."""
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
        # LCM flow-matching: takes the pipeline's custom sigmas AND the schedule flags.
        # Falls back to Euler when the installed diffusers does not expose it.
        try:
            from diffusers import FlowMatchLCMScheduler
            return FlowMatchLCMScheduler.from_config(config, **kw)
        except Exception as e:
            _log(f"sampler 'lcm' unavailable ({e}); falling back to euler")
    return FlowMatchEulerDiscreteScheduler.from_config(config, **kw)


def _apply_sampler(pipe):
    """Applies the current scheduler (SAMPLER x SCHEDULE) to a pipe. Checks compatibility
    (custom sigmas) and falls back to Euler/sgm_uniform when it fails -> never a crash at
    generation time."""
    if _BASE_SCHED_CONFIG is None:
        return
    from diffusers import FlowMatchEulerDiscreteScheduler
    try:
        sched = _build_scheduler(SAMPLER, SCHEDULE, _BASE_SCHED_CONFIG)
        if not _scheduler_accepts_sigmas(sched):
            raise ValueError(f"{type(sched).__name__} does not accept the custom sigmas of FLUX.2")
        pipe.scheduler = sched
        _dbg(f"sampler applied: {SAMPLER}/{SCHEDULE} -> {type(pipe.scheduler).__name__}")
    except Exception as e:
        _log(f"sampler '{SAMPLER}/{SCHEDULE}' incompatible ({e}); fallback Euler/sgm_uniform")
        try:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(_BASE_SCHED_CONFIG)
        except Exception:
            pass


# Changing the scheduler is NOT a local change: it lives on the SHARED pipe. A denoise
# loop already running keeps its own `timesteps` list but steps whatever `pipe.scheduler`
# points at by then. A fresh scheduler knows nothing of those timesteps and has no
# begin_index, so diffusers looks the current timestep up and finds nothing:
#   IndexError: index 0 is out of bounds for dimension 0 with size 0
#   (scheduling_flow_match_euler_discrete._init_step_index -> index_for_timestep)
# Met on crispz-krea2 on 2026-09-28: checkpoint switched, then "Apply CivitAI recommended
# settings" (which sets the sampler AND the schedule), then Generate. Gradio does not
# serialise the events of DIFFERENT listeners -- the very hole _GPU_LOCK exists for, except
# that these two setters were never put under it.
# So the swap never lands under a running generation: free lock -> applied at once; held
# lock -> only recorded, and the next get_pipe() applies it. Every generation path goes
# through get_pipe() with the lock held.
_SAMPLER_DIRTY = False


def _reapply_sampler_all():
    """Re-applies the current scheduler to every cached pipe (base + derived). Returns
    False when a generation holds the GPU: the change is recorded, not applied."""
    global _SAMPLER_DIRTY
    # A try-acquire, not a wait: blocking here would freeze the dropdown handler for the
    # whole render (up to a 30-image batch). RLock -> a call from a thread that ALREADY
    # holds the lock (the job queue restoring a snapshot between two jobs) goes through and
    # applies at once, which is correct: that thread is between two generations.
    if not _GPU_LOCK.acquire(blocking=False):
        _SAMPLER_DIRTY = True
        _log(f"sampler/schedule {SAMPLER}/{SCHEDULE}: applied on the NEXT run "
             f"(a generation is running; swapping it now would crash that render)")
        return False
    try:
        _SAMPLER_DIRTY = False
        for p in [_BASE_PIPE] + list(_DERIVED.values()):
            if p is not None:
                _apply_sampler(p)
    finally:
        _GPU_LOCK.release()
    return True


def _apply_sampler_if_dirty():
    """Applies a sampler/schedule change that arrived while a generation was running.
    Called by get_pipe(), i.e. by every generation path, with _GPU_LOCK held."""
    global _SAMPLER_DIRTY
    if not _SAMPLER_DIRTY:
        return
    _SAMPLER_DIRTY = False
    _dbg(f"applying the deferred sampler/schedule {SAMPLER}/{SCHEDULE}")
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            _apply_sampler(p)


def _sampler_status():
    """The label shown next to the two dropdowns. Says so when the change is only
    recorded: telling the user it is active while the render still uses the old one is
    exactly the confusion to avoid."""
    return (f"Sampler: {SAMPLER} / {SCHEDULE}"
            + (" — on the next run" if _SAMPLER_DIRTY else ""))


def set_sampler(name):
    """Changes the sampler (euler/unipc) and re-applies it to the cached pipes (no
    reload). No effect on the Omni pipe (its own scheduler)."""
    global SAMPLER
    name = (name or "euler").strip().lower()
    if name not in SAMPLER_CHOICES:
        name = "euler"
    if name != SAMPLER:
        SAMPLER = name
        _log(f"sampler -> {SAMPLER}")
        _reapply_sampler_all()
    return _sampler_status()


def set_schedule(name):
    """Changes the sigma schedule (sgm_uniform/beta/karras/exponential, alias 'simple'
    = sgm_uniform) and re-applies it to the cached pipes."""
    global SCHEDULE
    name = _norm_schedule(name)
    if name != SCHEDULE:
        SCHEDULE = name
        _log(f"schedule -> {SCHEDULE}")
        _reapply_sampler_all()
    return _sampler_status()


def _progress(frac, desc=""):
    if _PROGRESS is not None:
        try:
            _PROGRESS(min(1.0, max(0.0, float(frac))), desc)
        except Exception:
            pass


# ---- Model loading feedback (terminal + UI) ----
# from_pretrained is blocking and silent (the first load downloads from HF -> several
# minutes). So the load runs in a thread and every ~2s a terminal line plus the Gradio bar
# are refreshed (elapsed time + allocated VRAM). Config block "load_progress";
# enabled=false -> a direct load (no thread, zero cost).
_LOAD_CFG = CONFIG.get("load_progress") if isinstance(CONFIG.get("load_progress"), dict) else {}
LOAD_PROGRESS_ENABLED = bool(_LOAD_CFG.get("enabled", True))
_LOAD_TARGET_GB = float(_LOAD_CFG.get("target_vram_gb", 14.0))
_LOAD_HEARTBEAT = float(_LOAD_CFG.get("heartbeat_s", 2.0))


def _fmt_load(label, elapsed, vram_gb):
    """Loading progress text (pure, testable). VRAM > 0 -> the loading-into-memory
    phase; otherwise the download/disk-read phase."""
    if vram_gb > 0.05:
        return f"{label}... {elapsed:.0f}s | {vram_gb:.1f} GB in VRAM"
    return f"{label}... {elapsed:.0f}s (downloading / reading, first run only)"


def _load_pct(elapsed, vram_gb, target_gb=None):
    """An honest %: based on the allocated VRAM / target once the load into memory has
    started (capped at 0.95); during the download (VRAM~0) a small time-based bar."""
    target_gb = target_gb or _LOAD_TARGET_GB
    if vram_gb <= 0.05:
        return min(0.12, elapsed / 600.0)
    return min(0.95, vram_gb / max(1.0, float(target_gb)))


def _load_monitor(label, fn):
    """Runs fn() (a blocking load) in a thread and refreshes terminal + UI (time +
    VRAM) every ~2s. Returns fn's result (re-raises its exception)."""
    if not LOAD_PROGRESS_ENABLED:
        return fn()
    box = {}

    def _work():
        try:
            box["v"] = fn()
        except BaseException as e:   # noqa: BLE001 - it is re-raised in the main thread
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
    """Asks for a stop: halts the running denoise loop (pipe._interrupt) and the
    batch/tile loops (_STOP). Near-immediate (it stops at the next step)."""
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


@_gpu_exclusive
def set_zimage_model(repo_or_path):
    """Changes the Klein model. An HF repo / diffusers folder -> BASE_REPO.
    A single-file checkpoint (a Civitai .safetensors, a .gguf) -> a transformer override."""
    global BASE_REPO, ZIMAGE_TRANSFORMER
    if not repo_or_path:
        return
    if _is_single_file(repo_or_path):
        # A transformer-only change: NO free_vram -> _ensure_base will swap the
        # transformer alone (VAE + text encoder kept in VRAM).
        if repo_or_path != ZIMAGE_TRANSFORMER:
            ZIMAGE_TRANSFORMER = repo_or_path
            _log("Klein transformer (single-file) changed -> transformer swap on next run")
    elif repo_or_path != BASE_REPO:
        # The base repo changes: the VAE/encoder/tokenizer change too -> a full reload.
        BASE_REPO = repo_or_path
        # The repo's dimension is cached, FAILURES INCLUDED. A gated repo whose licence
        # had not been accepted yet therefore left a None stuck for the whole session:
        # the 4B/9B filter stayed off even after accepting the licence, until a restart.
        # Choosing that repo counts as "retry", so its entry is purged.
        _BASE_DIM_CACHE.pop(repo_or_path, None)
        free_vram()
        _log("Klein base repo changed -> will reload")


def set_zimage_transformer(path):
    """Sets (or removes with '' / None) the single-file transformer.

    Does NOT release the pipeline: with the same base repo, _ensure_base will only reload
    the transformer (_swap_transformer) and keep VAE + text encoder in VRAM.
"""
    global ZIMAGE_TRANSFORMER
    path = path or None
    if path != ZIMAGE_TRANSFORMER:
        ZIMAGE_TRANSFORMER = path
        _log(f"Klein transformer -> {path or '(base repo)'} "
             "-> transformer swap on next run (base components kept)")


# --- Replacement text encoder ---------------------------------------------------------
# FLUX.2 reads three INTERMEDIATE hidden states of the encoder (a context_embedder 3 x
# hidden wide): an encoder only fits if it has the same family, the same width and the same
# number of layers as the base repo's. An "abliterated" or fine-tuned Qwen3 of the same size
# plugs in as is. That is checked on the config, BEFORE reading 8 GB.
_ENCODER_WIDTH_FOR = {2560: "FLUX.2-klein-4B (Qwen3-4B)", 4096: "FLUX.2-klein-9B (Qwen3-8B)"}


def _split_hf_src(src):
    """'owner/repo/sub/folder' -> ('owner/repo', 'sub/folder'). The weights of an encoder
    published on HF often sit in a subfolder of the repo."""
    parts = [p for p in str(src).replace("\\", "/").split("/") if p]
    if len(parts) > 2:
        return "/".join(parts[:2]), "/".join(parts[2:])
    return str(src), None


def _enc_dims(cfg):
    """(width, layers, family) of a transformers config. The VLs keep the text part under
    'text_config'; T5 says d_model / num_layers."""
    c = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    h = c.get("hidden_size") or c.get("d_model")
    n = c.get("num_hidden_layers") or c.get("num_layers")
    return (int(h) if h else None, int(n) if n else None, cfg.get("model_type"))


def _base_text_encoder_config(base=None):
    """The config.json of the base repo's encoder, or None when unreadable."""
    base = (base or BASE_REPO or "").strip()
    try:
        cfg = os.path.join(base, "text_encoder", "config.json")
        if not os.path.isfile(cfg):
            from huggingface_hub import hf_hub_download
            try:
                cfg = hf_hub_download(base, "text_encoder/config.json", local_files_only=True)
            except Exception:
                cfg = hf_hub_download(base, "text_encoder/config.json")
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _dbg(f"cannot read {base}'s text encoder config: {e}")
        return None


def _text_encoder_source(src):
    """Locates the encoder `src`: (config, folder or repo, subfolder) or None.
    A local folder: config.json at the root or inside text_encoder/. An HF repo: the same,
    or the subfolder named in the id."""
    src = (src or "").strip()
    if not src:
        return None
    if os.path.isdir(src):
        for sub in (None, "text_encoder"):
            p = os.path.join(src, sub, "config.json") if sub else os.path.join(src, "config.json")
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        return json.load(f), src, sub
                except Exception:
                    return None
        return None
    if os.path.exists(src) or _looks_single_file(src) or "\\" in src or os.path.isabs(src):
        return None
    repo, sub0 = _split_hf_src(src)
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    for sub in ([sub0] if sub0 else [None, "text_encoder"]):
        rel = f"{sub}/config.json" if sub else "config.json"
        for local in (True, False):          # the cache first: works offline
            try:
                p = hf_hub_download(repo, rel, local_files_only=local)
                with open(p, encoding="utf-8") as f:
                    return json.load(f), repo, sub
            except Exception:
                continue
    return None


def _encoder_label(src):
    """A readable name for an encoder: the FOLDER's name -- never the path, which would
    end up in shared PNGs along with the Windows session name -- or the HF repo id."""
    src = (src or "").strip()
    if not src:
        return ""
    if os.path.isabs(src) or os.path.exists(src) or "\\" in src:
        parts = [p for p in src.replace("\\", "/").split("/") if p]
        if len(parts) >= 2 and parts[-1] == "text_encoder":
            return parts[-2]
        return parts[-1] if parts else src
    return src


def _text_encoder_problem(src, base=None):
    """Why `src` should be refused as the encoder of repo `base`, or None when it fits."""
    src = (src or "").strip()
    if not src:
        return None
    if src.lower().endswith(".gguf"):
        return ("a GGUF text encoder is a ComfyUI / llama.cpp file; this app loads the "
                "transformers folder (config.json + .safetensors)")
    if os.path.isfile(src) or _looks_single_file(src):
        return ("a single file carries no config.json; point to the FOLDER that holds "
                "config.json and the weights")
    found = _text_encoder_source(src)
    if found is None:
        return ("no config.json found, neither at its root nor in text_encoder/"
                if os.path.isdir(src) else
                "neither a folder on this machine nor a readable Hugging Face repo")
    ref_cfg = _base_text_encoder_config(base)
    if ref_cfg is None:
        return None                      # nothing to compare: the load will decide
    (h, n, t), (rh, rn, rt) = _enc_dims(found[0]), _enc_dims(ref_cfg)
    b = (base or BASE_REPO)
    if t and rt and t != rt:
        return f"a '{t}' model, and {b} uses a '{rt}' text encoder"
    if h and rh and h != rh:
        made = _ENCODER_WIDTH_FOR.get(h)
        return (f"hidden size {h}, and {b}'s encoder is {rh} wide: the transformer "
                f"cannot read its embeddings" + (f" (this is an encoder for {made})" if made else ""))
    if n and rn and n != rn:
        return f"{n} layers, and {b}'s encoder has {rn}"
    return None


def _encoder_class(base=None):
    """The encoder's transformers class, read from the base repo's model_index.json
    (Qwen3ForCausalLM here): the same one diffusers would have loaded."""
    base = (base or BASE_REPO or "").strip()
    try:
        p = os.path.join(base, "model_index.json")
        if not os.path.isfile(p):
            from huggingface_hub import hf_hub_download
            try:
                p = hf_hub_download(base, "model_index.json", local_files_only=True)
            except Exception:
                p = hf_hub_download(base, "model_index.json")
        with open(p, encoding="utf-8") as f:
            lib, cls = json.load(f)["text_encoder"]
        import importlib
        return getattr(importlib.import_module(lib), cls)
    except Exception as e:
        raise RuntimeError(f"cannot tell which class {base}'s text encoder uses "
                           f"({type(e).__name__}: {e})") from e


def _load_text_encoder(src, base=None):
    """Loads the encoder `src` in DTYPE, with the base repo's class."""
    found = _text_encoder_source(src)
    if found is None:
        raise RuntimeError(f"{src}: no config.json")
    _cfg, where, sub = found
    kw = {"torch_dtype": DTYPE}
    if sub:
        kw["subfolder"] = sub
    return _encoder_class(base).from_pretrained(where, **kw)


def list_text_encoders():
    """Encoder folders offered in the Models tab: the subfolders with a config.json inside
    `text_encoders_dir`, or inside text_encoders / text_encoder / clip next to the
    checkpoints folder or its parent (ComfyUI and Forge conventions)."""
    roots = [TEXT_ENCODERS_DIR] if TEXT_ENCODERS_DIR else []
    here = os.path.abspath(CHECKPOINTS_DIR or ".")
    for up in (os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        roots += [os.path.join(up, n) for n in ("text_encoders", "text_encoder", "clip")]
    out = []
    for r in roots:
        try:
            names = sorted(os.listdir(r))
        except OSError:
            continue
        for d in names:
            p = os.path.join(r, d)
            if p in out or not os.path.isdir(p):
                continue
            if (os.path.isfile(os.path.join(p, "config.json"))
                    or os.path.isfile(os.path.join(p, "text_encoder", "config.json"))):
                out.append(p)
    return out



def _hf_cache_dir():
    """The Hugging Face cache folder (follows HF_HUB_CACHE / HF_HOME), or None."""
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except Exception:
        return None


def list_cached_text_encoders(base=None):
    """COMPATIBLE encoders already downloaded in the Hugging Face cache, as (name, HF id).

    An encoder downloaded from Hugging Face lives in that cache, not in a text_encoders
    folder: without this sweep the Models tab list only showed "Default" (caught on
    2026-09-10 with the huihui and ponpoke encoders downloaded). Offered: the repos that
    are NOT diffusers pipelines (no model_index.json), one of whose configs -- at the root
    or in a subfolder -- matches the base repo's encoder (family, width, layers), and whose
    weights are there. The value is the HF id: that is what ends up in the metadata,
    readable, instead of the snapshot's hash.
"""
    ref_cfg = _base_text_encoder_config(base)
    if not ref_cfg:
        return []
    ref = _enc_dims(ref_cfg)
    return [(hid, hid) for hid, cfg in _scan_cached_encoders() if _enc_dims(cfg) == ref]


def cached_text_encoder_mismatches(base=None):
    """Encoders in the HF cache of the SAME family but of another size than the base repo's
    -- e.g. a Qwen3-4B when klein 9B is loaded. Hidden from the list (they would be
    refused), they are named next to it: otherwise one believes the download failed.
    ([(HF id, width)], expected width)."""
    ref_cfg = _base_text_encoder_config(base)
    if not ref_cfg:
        return [], None
    rh, rn, rt = _enc_dims(ref_cfg)
    out = []
    for hid, cfg in _scan_cached_encoders():
        h, n, t = _enc_dims(cfg)
        if t == rt and (h, n) != (rh, rn):
            out.append((hid, h))
    return out, rh


def _scan_cached_encoders():
    """[(HF id, config)] from the Hugging Face cache: repos that are NOT diffusers
    pipelines, one of whose configs -- at the root or in a subfolder -- has its weights next
    to it (the most recent revision)."""
    root = _hf_cache_dir()
    if not root or not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        if not d.startswith("models--"):
            continue
        repo = d[len("models--"):].replace("--", "/", 1)
        snaps = os.path.join(root, d, "snapshots")
        try:
            revs = sorted(os.listdir(snaps),
                          key=lambda r: os.path.getmtime(os.path.join(snaps, r)), reverse=True)
        except OSError:
            continue
        if not revs:
            continue
        snap = os.path.join(snaps, revs[0])         # la revision la plus recente
        if os.path.isfile(os.path.join(snap, "model_index.json")):
            continue                                 # a diffusers pipeline, not an encoder
        try:
            subs = [""] + sorted(s for s in os.listdir(snap) if os.path.isdir(os.path.join(snap, s)))
        except OSError:
            continue
        for s in subs:
            p = os.path.join(snap, s) if s else snap
            try:
                with open(os.path.join(p, "config.json"), encoding="utf-8") as f:
                    cfg = json.load(f)
                if not isinstance(cfg, dict):
                    continue
                if not any(fn.endswith(".safetensors") for fn in os.listdir(p)):
                    continue                         # the config alone, the weights are not downloaded
            except Exception:
                continue
            out.append((f"{repo}/{s}" if s else repo, cfg))
    return out

@_gpu_exclusive
def set_text_encoder(src):
    """Picks the text encoder ('' = the base repo's). A change RELEASES the pipeline --
    the encoder loads with it, no hot swap under the offload hooks -- and free_vram empties
    the embedding cache, which the previous one computed."""
    global TEXT_ENCODER
    src = (src or "").strip()
    if src == TEXT_ENCODER:
        return
    TEXT_ENCODER = src
    free_vram()
    _log(f"text encoder -> {_encoder_label(src) or '(base repo)'} -> full reload on next run")


def _safetensors_header(path):
    """The JSON header of a .safetensors (tensor names/dtypes/shapes, NEVER the
    weights) -- a few hundred KB read at most, even on a 12 GB file."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(min(n, 10_000_000)).decode("utf-8", "ignore"))


# --- FLUX.2 variant: 4B or 9B? ---------------------------------------------------
# Both share the architecture AND the tensor names: the architecture guard lets both
# through. Only the HIDDEN DIMENSION separates them, and a 9B loaded into a 4B pipeline
# blows up after reading gigabytes, on an unreadable diffusers message ("expected shape
# [18432, 3072], but got [24576, 4096]"). So it is decided from the header.
# The key carries '.lin.' in the original layout (ComfyUI) and '.linear.' in the diffusers
# one. Each signature comes with its structural out/in RATIO, which holds for both
# variants: double_stream 18432/3072 == 24576/4096 == 6, single_stream 9216/3072 ==
# 12288/4096 == 3. Requiring that ratio avoids taking any tensor carrying the right name
# (a test fixture, a truncated file) for a hidden dimension.
_FLUX2_DIM_SIGS = (("double_stream_modulation_img.lin.weight", 6),
                   ("double_stream_modulation_img.linear.weight", 6),
                   ("double_stream_modulation_txt.lin.weight", 6),
                   ("single_stream_modulation.lin.weight", 3),
                   ("single_stream_modulation.linear.weight", 3))
_FLUX2_VARIANTS = {3072: "4B", 4096: "9B"}
_BASE_DIM_CACHE = {}


def _flux2_hidden_dim_from_shapes(items):
    """Hidden dimension from a set of (name, shape). None when no signature is recognised."""
    for name, shape in items:
        for sig, ratio in _FLUX2_DIM_SIGS:
            if not name.endswith(sig) or shape is None or len(shape) != 2:
                continue
            out, dim = int(shape[0]), int(shape[1])
            if dim > 0 and out == dim * ratio:
                return dim
    return None


def _flux2_hidden_dim(path):
    """Hidden dimension of a single-file FLUX.2 transformer, read from the HEADER alone."""
    try:
        hdr = _safetensors_header(path)
    except Exception:
        return None
    return _flux2_hidden_dim_from_shapes(
        (k, v.get("shape")) for k, v in hdr.items() if k != "__metadata__")


def _lora_side(key):
    """'A' (input projection), 'B' (output projection) or None.

    Recognised by SEGMENT rather than by suffix: peft inserts the adapter's name
    ('...lora_A.default.weight'), and a frozen suffix misses that case -- seen for real on
    a LoRA from the library. The three living dialects are covered: lora_A/lora_B (peft),
    lora_down/lora_up and lora.down/lora.up.
"""
    parts = key.split(".")
    for i, seg in enumerate(parts):
        if seg in ("lora_A", "lora_down"):
            return "A"
        if seg in ("lora_B", "lora_up"):
            return "B"
        if seg == "lora" and i + 1 < len(parts):
            if parts[i + 1] == "down":
                return "A"
            if parts[i + 1] == "up":
                return "B"
    return None


def _flux2_lora_hidden_dim(path):
    """Hidden dimension of the model a LoRA was trained on, from the header alone.

    A LoRA holds no weight of the model, but its two matrices keep its trace: lora_A
    projects FROM the layer's input ([rank, input]), lora_B TOWARDS its output
    ([output, rank]). Wherever that input -- or that output -- IS the hidden dimension,
    the shape gives it away, without reading a single weight.

    NO layer is named. The first version listed the suffixes of the diffusers layout
    ('attn.to_q.lora_A.weight'...) and believed itself complete: on a real library of 70
    klein LoRAs it recognised exactly one. The others are in the original FLUX layout
    ('diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight') -- so the guard was
    inert precisely where it was needed. ALL the matrices are counted now and only the
    values declared in _FLUX2_VARIANTS are kept: the derived dimensions (qkv at 3x, mlp at
    4x) are not in there and drop out by themselves.

    None at the slightest doubt: discarding a valid LoRA would be worse than the error
    message this replaces.
"""
    try:
        hdr = _safetensors_header(path)
    except Exception:
        return None
    seen = {}
    for k, v in hdr.items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        shape = v.get("shape") or []
        if len(shape) != 2:
            continue
        side = _lora_side(k)
        if side is None:
            continue
        d = int(shape[1] if side == "A" else shape[0])
        if d in _FLUX2_VARIANTS:
            seen[d] = seen.get(d, 0) + 1
    return max(seen, key=seen.get) if seen else None


def _lora_variant_refusal(dim, base=None):
    """The named refusal of a LoRA trained for the OTHER variant. Without it, peft dumps
    forty lines of 'size mismatch ... torch.Size([27648, 128]) vs
    torch.Size([36864, 128])' where nobody reads that 27648 = 9 x 3072 and therefore 4B."""
    want = _base_hidden_dim(base)
    return (f"LoRA trained for {_variant_name(dim)}, and this build runs "
            f"{_variant_name(want)}. Its matrices carry the hidden size of the model "
            f"it was trained on ({dim} against {want}), so peft cannot fit them. To "
            f"use it, " + _variant_fix(dim))


def _base_hidden_dim(base=None):
    """The dimension the current base repo expects, read from transformer/config.json
    (hidden = attention_head_dim * num_attention_heads). None when undeterminable -- in
    that case nothing is filtered: better to try than to discard a valid model."""
    base = (base or BASE_REPO or "").strip()
    if not base:
        return None
    if base in _BASE_DIM_CACHE:
        return _BASE_DIM_CACHE[base]
    dim = None
    try:
        cfg = os.path.join(base, "transformer", "config.json")
        if not os.path.isfile(cfg):
            from huggingface_hub import hf_hub_download
            try:
                cfg = hf_hub_download(base, "transformer/config.json", local_files_only=True)
            except Exception:
                cfg = hf_hub_download(base, "transformer/config.json")
        d = json.load(open(cfg, encoding="utf-8"))
        hd, nh = d.get("attention_head_dim"), d.get("num_attention_heads")
        if hd and nh:
            dim = int(hd) * int(nh)
    except Exception as e:
        # Degradation announced: without the base repo's dimension, the 4B/9B sorting
        # of checkpoints is OFF (house rule: never discard on a doubt). The most frequent
        # cause on the 9B: a gated repo, licence not accepted or token missing -> the
        # transformer's config cannot even be read.
        _log(f"cannot read {base}'s transformer config ({type(e).__name__}: {e}) -> the "
             f"4B/9B checkpoint filter is OFF for this base: every single-file "
             f"checkpoint is listed, and a wrong-variant one will fail at load time.")
    _BASE_DIM_CACHE[base] = dim
    return dim


def _variant_name(dim):
    """'FLUX.2-klein-4B' / '-9B', or the raw dimension when the variant is unknown."""
    v = _FLUX2_VARIANTS.get(dim)
    return f"FLUX.2-klein-{v}" if v else f"hidden dim {dim}"


def _flux2_variant_mismatch(dim, base=None):
    """A SHORT reason when `dim` does not match the base repo, otherwise None.

    Short on purpose: the reason is repeated once per discarded file, and a library can
    hold dozens of them. The instructions (which key to change, the 9B's licence) are
    given ONCE per listing, by _variant_skip_summary.
"""
    if not dim:
        return None
    want = _base_hidden_dim(base)
    if not want or dim == want:
        return None
    return f"{_variant_name(dim)}, and this build runs {_variant_name(want)}"


def _variant_repo(dim):
    """This variant's official base repo, or None when it is unknown.
    Naming the exact repo beats "the matching repo": it is the value to pick in the
    dropdown, word for word."""
    v = _FLUX2_VARIANTS.get(dim)
    return f"black-forest-labs/FLUX.2-klein-{v}" if v else None


def _variant_fix(dim):
    """What to do to use this variant. The dropdown first: that is where the refusal
    comes from, and sending someone to a config file when a menu does the job is
    sending them to the wrong place."""
    repo = _variant_repo(dim)
    if not repo:
        return f"point '{CFG_MODEL_KEY}' at the matching base repo."
    line = (f"switch the base model to {repo} (the 'Klein checkpoint' dropdown, or "
            f"config '{CFG_MODEL_KEY}').")
    if _FLUX2_VARIANTS.get(dim) == "9B":
        line += (f" Heads-up: that repo is NON-COMMERCIAL (the 4B is Apache-2.0) and "
                 f"GATED - accept its licence at https://huggingface.co/{repo} first, "
                 f"and it needs ~29 GB of VRAM. See FORK.md.")
    return line


def _variant_note(dim):
    """The licence reminder, when the discarded variant is the 9B."""
    if _FLUX2_VARIANTS.get(dim) == "9B":
        return (" Note: FLUX.2-klein-9B is NON-COMMERCIAL, unlike the 4B "
                "(Apache-2.0) - see FORK.md.")
    return ""


def _variant_skip_summary(n, dim, base=None):
    """The instructions, once only for the n discarded files."""
    want = _base_hidden_dim(base)
    return (f"{n} checkpoint(s) skipped: they are {_variant_name(dim)} builds and this "
            f"install runs {_variant_name(want)} ({base or BASE_REPO}). To use them, "
            + _variant_fix(dim))


def _variant_refusal(dim, base=None):
    """The same instructions, for ONE file someone has just tried to select."""
    want = _base_hidden_dim(base)
    return (f"it is a {_variant_name(dim)} build and this install runs "
            f"{_variant_name(want)} ({base or BASE_REPO}). To use it, "
            + _variant_fix(dim))


# Tensor suffixes specific to LyCORIS. LoKr factorises the update as a Kronecker
# product (w1 (x) w2), LoHa as a Hadamard product: neither is a LoRA in peft's sense, and
# diffusers has NO conversion for them (checked: not a single occurrence of 'lokr' in
# loaders/lora_conversion_utils.py).
_LYCORIS_SUFFIXES = ("lokr_", "hada_")


def _lycoris_algo_from_header(hdr):
    """'LoKr', 'LoHa' or None, from a header already read."""
    suf = [k.rsplit(".", 1)[-1] for k in hdr if k != "__metadata__"]
    if sum(1 for s in suf if s.startswith("hada_")) >= 4:
        return "LoHa"
    if sum(1 for s in suf if s.startswith("lokr_")) >= 4:
        return "LoKr"
    return None


_LYCORIS_CACHE = {}


def _lycoris_algo(path):
    """The LyCORIS algorithm of a file, read from the header alone, memoised on
    (path, size, mtime): _apply_loras is called ON EVERY generation, and re-reading the
    header of a one-gigabyte file on a network drive for every image would be paying dearly
    for an answer that does not change."""
    try:
        k = _file_key(path)
    except OSError:
        return None
    if k not in _LYCORIS_CACHE:
        try:
            _LYCORIS_CACHE[k] = _lycoris_algo_from_header(_safetensors_header(path))
        except Exception:
            _LYCORIS_CACHE[k] = None
    return _LYCORIS_CACHE[k]


def _lycoris_reason(hdr):
    """The named refusal for a LyCORIS found where it does not belong."""
    algo = _lycoris_algo_from_header(hdr) or "adapter"
    if algo == "LoKr":
        return ("LyCORIS LoKr, not a checkpoint - it IS supported, but as an adapter: "
                "move it to the LoRA folder and pick it in Models > LoRA, where it is "
                "merged into the weights at load")
    return (f"LyCORIS {algo}, neither a LoRA nor a checkpoint - only LoKr is supported "
            f"here. Use a version already merged into a base model, or merge it "
            f"yourself with LyCORIS/sd-scripts first")


def _lora_unsupported(path):
    """A reason (str) when this file cannot be applied as a PEFT adapter, otherwise
    None. A LoKr returns None: it IS supported, by merging (_merge_lokr), and it is
    removed upstream from the set handed to peft. A LoHa stays refused by name -- without
    that it goes as is into load_lora_weights, which recognises none of its keys, applies
    NOTHING and says nothing."""
    # Wrong variant (a 4B LoRA on a 9B base, or the reverse). First: it is the common
    # case, and the only one whose raw failure is unreadable.
    bad = _flux2_variant_mismatch(_flux2_lora_hidden_dim(path))
    if bad:
        return _lora_variant_refusal(_flux2_lora_hidden_dim(path))
    algo = _lycoris_algo(path)
    if algo and algo != "LoKr":
        return (f"LyCORIS {algo} - only LoKr is supported here; peft recognises none "
                f"of its Hadamard factors and would apply nothing, silently")
    # Quantized. This app's dequant loader serves ONLY the transformer
    # (_safetensors_dequant is called from _load_transformer alone): a quantized LoRA would
    # go as is into load_lora_weights, where its 'weight_scale' tensors are not known LoRA
    # keys -- so they are ignored -- and where its fp8/int8 weights would be cast to bf16
    # WITHOUT their scale. Result: values several orders of magnitude too small, that is a
    # LoRA doing nothing, without the slightest message. The same trap as FP4 and LyCORIS,
    # through the same door.
    try:
        hdr = _safetensors_header(path)
        fp4 = sorted(f for f in _quant_metadata_formats(hdr) if "fp4" in f)
        if fp4 or any(str(v.get("dtype", "")).upper().startswith("F4")
                      for k, v in hdr.items()
                      if k != "__metadata__" and isinstance(v, dict)):
            return (f"{(fp4[0].upper() if fp4 else 'FP4')} (4-bit) LoRA - there is no "
                    f"FP4 path here, for adapters or anything else. Take the bf16 "
                    f"download of the same LoRA")
    except Exception:
        pass
    dq = _safetensors_dequant(path)
    if dq:
        return (f"{dq} LoRA - this build dequantizes the TRANSFORMER only, never an "
                f"adapter. peft would not recognise its scale tensors, would drop "
                f"them, and would apply weights orders of magnitude too small - a "
                f"LoRA that does nothing, silently. Take the bf16 download of the "
                f"same LoRA")
    return None


def _quant_metadata_formats(hdr):
    """The quantization formats declared in __metadata__._quantization_metadata
    (ComfyUI, NVIDIA ModelOpt), lowercased. An empty set when the file declares none. It is
    the most reliable source: it names the format even when the safetensors dtype does not
    distinguish it (packed 4-bit presents itself as U8)."""
    try:
        raw = (hdr.get("__metadata__") or {}).get("_quantization_metadata")
        if not raw:
            return set()
        layers = json.loads(raw).get("layers") or {}
        return {str(v.get("format", "")).lower()
                for v in layers.values() if isinstance(v, dict)}
    except Exception:
        return set()


# Signature of a VAE (autoencoder) filed among the checkpoints: top-level blocks,
# then markers SPECIFIC to a VAE -- a T5 text encoder also has 'encoder.' keys, but never a
# post_quant_conv nor a decoder.conv_in.
_VAE_TOP = ("encoder", "decoder", "quant_conv", "post_quant_conv", "bn")
_VAE_MARKERS = ("post_quant_conv", "quant_conv", "decoder.conv_in", "decoder.mid")
# Everything that betrays a diffusion transformer, ComfyUI prefix or not. Broader than
# the historical `dit_keys` counter ('transformer_blocks', 'img_in'), which does not see
# 'model.diffusion_model.double_blocks.*': reused here, it would have refused as VAEs the
# library's two all-in-one bundles, which load perfectly well.
_DIT_MARKERS = ("transformer_blocks", "double_blocks", "single_blocks", "x_embedder",
                "context_embedder", "img_in", "txt_in")


def _safetensors_unsupported(path):
    """Returns a reason (str) when the .safetensors is NOT loadable, otherwise None.
    Only reads the header (fast). Three cases stay unsupported:
      - a LoRA file filed in the checkpoints folder (kohya/peft keys)
      - SVDQuant / Nunchaku (tensors named '*.qweight'): pre-quantized INT4 weights that
        require the nunchaku runtime (dedicated kernels), not dequantizable here.
      - a bare VAE (autoencoder, often named 'diffusion_pytorch_model'): not a
        transformer, the pipeline takes its VAE from the base repo.
      - NVFP4 / MXFP4 (4-bit): neither dequant (the 4-bit packing and the scale
        convention are their own) nor runtime (TensorRT/ModelOpt). Naming them is
        essential: an unrecognised FP4 has no F8 dtype, so it ESCAPES the dequant loader
        and goes down the normal bf16 path, where it gives at best an unreadable diffusers
        error, at worst a flat image.
    ComfyUI-style 'scaled' FP8 / INT8 are NO LONGER rejected: they go through the dequant
    loader (_safetensors_dequant + _load_dequant_state_dict).
"""
    bad = _flux2_variant_mismatch(_flux2_hidden_dim(path))
    if bad:
        return bad
    try:
        hdr = _safetensors_header(path)
        has_qweight = False
        has_fp4 = False
        lora_keys = 0
        lycoris_keys = 0
        te_keys = 0
        dit_keys = 0
        dit_any = 0
        vae_keys = 0
        vae_marker = False
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            if k.endswith(".qweight"):
                has_qweight = True
            # F4_E2M1 (safetensors >= 0.5). An FP4 packed in pairs sometimes declares
            # itself as U8: the dtype alone is not enough, hence the cross-check with the
            # quantization metadata below.
            if str(v.get("dtype", "")).upper().startswith("F4"):
                has_fp4 = True
            # A Qwen2.5-VL text encoder (the ComfyUI file
            # 'qwen_2.5_vl_7b_fp8_scaled'): LLM layers + a vision tower, never any
            # diffusion block.
            if k.startswith(("model.layers.", "visual.", "lm_head.", "model.embed_tokens",
                             "language_model.", "model.language_model.")):
                te_keys += 1
            if "transformer_blocks" in k or k.startswith(("img_in", "txt_in")):
                dit_keys += 1
            if any(m in k for m in _DIT_MARKERS):
                dit_any += 1
            kk = k[4:] if k.startswith("vae.") else k
            if kk.split(".", 1)[0] in _VAE_TOP:
                vae_keys += 1
                if kk.startswith(_VAE_MARKERS):
                    vae_marker = True
            if (".lora_down." in k or ".lora_up." in k or ".lora_A." in k
                    or ".lora_B." in k or k.startswith(("lora_unet_", "lora_te"))):
                lora_keys += 1
            if k.rsplit(".", 1)[-1].startswith(_LYCORIS_SUFFIXES):
                lycoris_keys += 1
        # A LyCORIS (LoKr/LoHa) filed with the checkpoints. None of the guards below
        # saw it: its keys carry NEITHER '.lora_A/B' NOR the 'lora_unet_' prefix
        # (ai-toolkit writes 'diffusion_model.<module>.lokr_w1'), so it passed for a
        # checkpoint and went into from_single_file.
        if lycoris_keys >= 4:
            return _lycoris_reason(hdr)
        # A LoRA file filed in the checkpoints folder (a classic mistake): loading it
        # as a transformer sends diffusers looking for a default config (SD1.5) -> a 404
        # 'stable-diffusion-v1-5 does not appear to have a file named config.json'.
        if lora_keys >= 4:
            return "LoRA file, not a checkpoint - move it to the LoRA folder and pick it in Models > LoRA"
        # A text encoder filed with the checkpoints (a Civitai 'text encoder'
        # download): it is not an image model, dequantizing it would waste ~15 GB of cache
        # and loading it as a transformer would fail. The pipe takes its encoder from the
        # base repo.
        if te_keys >= 4 and dit_keys == 0:
            return ("text encoder (Qwen3), not an image model - the pipeline takes its "
                    "text encoder from the base repo; nothing to do with this file")
        # A VAE (autoencoder) filed with the checkpoints. It often carries the generic
        # name 'diffusion_pytorch_model.safetensors', the one diffusers gives to EVERY
        # component -- hence the confusion with a transformer. Loaded as one, no weight
        # finds its place, everything stays on 'meta', and generation crashes on "Cannot
        # copy out of meta tensor". An all-in-one bundle (transformer + VAE) is NOT
        # concerned: it carries transformer keys, and the guard requires there be none.
        if vae_keys >= 8 and vae_marker and dit_any == 0:
            return ("VAE (autoencoder), not a transformer - the pipeline takes its VAE "
                    "from the base repo, so this file does nothing here; move it out of "
                    "the checkpoints folder")
        # '*.qweight' = pre-quantized weights (SVDQuant/Nunchaku, GPTQ-like). A clear signal:
        # a normal BF16/FP16 checkpoint never has a 'qweight'.
        if has_qweight:
            return "SVDQuant/Nunchaku INT4"
        # 4-bit (NVFP4/MXFP4). The DECLARED format is named rather than a generic
        # "FP4": a Civitai page often offers the same model in bf16, fp8 and fp4, and
        # knowing which one you hold tells you which one to download again.
        fp4 = sorted(f for f in _quant_metadata_formats(hdr) if "fp4" in f)
        if fp4 or has_fp4:
            what = fp4[0].upper() if fp4 else "FP4"
            return (f"{what} (4-bit) - this build has no FP4 path, neither dequant nor "
                    f"runtime (that needs TensorRT/ModelOpt). Take the fp8 or bf16 "
                    f"version of the same model instead")
    except Exception:
        pass
    return None


def _safetensors_dequant(path):
    """Returns the ComfyUI quantization scheme to dequantize at load time
    ('FP8', 'FP8 scaled' or 'INT8 scaled'), otherwise None (BF16/FP16 -> the normal path).
    The ComfyUI 'scaled' format seen on Civitai checkpoints:
      X.weight (F8_E4M3 or I8) + X.weight_scale (F32, scalar or per row [out,1])
      + X.comfy_quant (a small U8 descriptor blob, to be dropped).
    NB: an AIO bundle whose text encoder ALONE is quantized (BF16 transformer) also
    triggers -> the dequant loader filters the transformer and leaves it untouched.
    U8 alone does not trigger: 'comfy_quant' blobs are U8 in healthy files.
"""
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


# Key markers of the FLUX.2 transformer (original layout OR ComfyUI prefix): used by
# the dequant loader to refuse a quantized checkpoint of ANOTHER architecture (it would
# load incoherent weights).
# Key markers specific to Flux2Transformer2DModel (read off the transformer of
# FLUX.2-klein-4B: 169 tensors, prefixes transformer_blocks / single_transformer_blocks /
# x_embedder / context_embedder / double_stream_modulation_* / time_guidance_embed).
# The name _QWEN_KEY_MARKERS is kept to limit the conflict surface when merging from
# qwen/main -- only the CONTENT changes.
_QWEN_KEY_MARKERS = ("single_transformer_blocks.", "double_stream_modulation",
                     "x_embedder", "context_embedder")

# The ComfyUI/LDM prefix of single-file diffusion checkpoints. diffusers 0.39 maps
# Flux2Transformer2DModel with an IDENTITY function (no key conversion at all): the state
# dict must therefore arrive IN THE DIFFUSERS LAYOUT, prefix stripped. Otherwise every key
# is "unexpected", no weight is loaded, the model stays on 'meta' and dispatch_model breaks
# on "Cannot copy out of meta tensor; no data!".
_COMFY_PREFIX = "model.diffusion_model."


# ----------------------------------------------------------------------------
# Disk cache of dequantized transformers (ComfyUI FP8/INT8 -> bf16), carried over
# from crispz-studio. A dequant reads and converts the whole file (minutes on a
# HDD); the bf16 is written ONCE here and later loads become a normal
# single-file one (seconds). The KEY is the ORIGINAL file (path+size+mtime):
# deleting this cache is always safe, it rebuilds on demand.
# ----------------------------------------------------------------------------
import hashlib as _dqhash

_DQ_CACHE_CFG = str(CONFIG.get("dequant_cache", "auto") or "auto").strip()
try:
    DEQUANT_CACHE_MAX_GB = float(CONFIG.get("dequant_cache_max_gb", 60) or 0)
except Exception:
    DEQUANT_CACHE_MAX_GB = 60.0


def _file_key(path):
    """A file's stable, cheap identity: (absolute path, size, mtime)."""
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, int(st.st_mtime))


def _dequant_cache_dir():
    """The dequant cache folder, created on demand. None = cache disabled."""
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


def _dequant_cache_path(src, legacy=False):
    """Path of the cached bf16 for a source checkpoint. The key includes size+mtime: a
    replaced file (same name) never reuses the old cache.

    A file already stored at scale (_source_prescaled) changes key: its old cache was
    written by the loader that applied the scale wrongly, so it holds wrong weights. The
    other files keep their key -- and their cache, which costs minutes per model to
    rebuild. legacy=True returns the old key.
"""
    d = _dequant_cache_dir()
    if not d:
        return None
    try:
        p, size, mtime = _file_key(src)
    except OSError:
        return None
    tag = "bf16-v2"
    if not legacy and _source_prescaled(src):
        tag = "bf16-v2-prescaled"
    h = _dqhash.sha1(
        f"{p.lower()}|{size}|{mtime}|{tag}".encode("utf-8")).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(src))[0][:48]
    return os.path.join(d, f"{base}.{h}.safetensors")


def _dequant_cache_prune(keep=None):
    """Caps the cache (dequant_cache_max_gb, 0 = unlimited): deletes the least recently
    USED files (atime, else mtime) until it is back under the threshold."""
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
        for _t, size, fp in sorted(files):          # oldest access first
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
    """Writes the dequantized state dict into the cache (best effort: any error is
    ignored, the current load already has the dict in memory). Atomic write through a
    renamed .tmp -> an interruption never leaves a truncated cache."""
    dst = _dequant_cache_path(src)
    if not dst:
        return
    try:
        from safetensors.torch import save_file
        t0 = time.time()
        tmp = dst + ".tmp"
        # contiguous(): safetensors refuses non-contiguous views (they come from the
        # dequant slices); an implicit clone, and we are in RAM already.
        save_file({k: v.contiguous() for k, v in sd.items()}, tmp)
        os.replace(tmp, dst)
        gb = os.path.getsize(dst) / 1024**3
        _log(f"dequant cache: saved {gb:.1f} GB in {time.time() - t0:.1f}s "
             f"-> next load of this checkpoint skips the dequant")
        # THIS file's old cache, if it was written under the old key (wrong weights,
        # see _dequant_cache_path): it has been replaced, so it is deleted -- otherwise it
        # sleeps on the disk until the cache ceiling evicts it.
        old = _dequant_cache_path(src, legacy=True)
        if old and os.path.abspath(old) != os.path.abspath(dst) and os.path.isfile(old):
            try:
                ogb = os.path.getsize(old) / 1024**3
                os.remove(old)
                _log(f"dequant cache: removed the stale {os.path.basename(old)} "
                     f"({ogb:.1f} GB, written by the loader that applied the scale twice)")
            except OSError as e:
                _dbg(f"stale dequant cache not removed {old}: {e}")
        _dequant_cache_prune(keep=dst)
    except Exception as e:
        _log(f"dequant cache: not saved ({e})")
        try:
            os.remove(dst + ".tmp")
        except OSError:
            pass


def _hadamard_ortho(n):
    """The comfy-quants ConvRot 'regular hadamard' matrix -- CAREFUL, this is NOT
    Sylvester's construction: the base is that precise H4, extended by Kronecker products
    up to n (a power of 4), then normalised by 1/sqrt(n). Orthonormal AND symmetric -> the
    reconstruction simply multiplies by the same matrix again.
    (Checked against src/comfy_quants/formats/convrot.py; with a Sylvester the correlation
    to the base weights drops to ~0 -> pure noise.)
"""
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]])
    H = h4
    while H.shape[0] < n:
        H = torch.kron(H, h4)
    if H.shape[0] != n:
        raise ValueError(f"convrot groupsize {n} is not a power of 4")
    return H / (float(n) ** 0.5)


def _safetensors_comfy_prefixed(path):
    """True when the .safetensors is in the ComfyUI layout ('model.diffusion_model.*').
    Only reads the header. Such a file can NOT go as is into from_single_file (identity
    mapping on the diffusers side, see _COMFY_PREFIX)."""
    try:
        return any(k.startswith(_COMFY_PREFIX)
                   for k in _safetensors_header(path) if k != "__metadata__")
    except Exception:
        return False


# Range of each 8-bit integer/float format: the largest stored value a QUANTIZED
# weight (weight / scale) can reach.
_QUANT_RANGE = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0, torch.int8: 127.0}


def _stored_at_scale(t, s, qdtype, cfg=None):
    """True when the stored weights are ALREADY at their real scale, a weight_scale being
    supplied on top -- not to be applied.

    A normal 'scaled' FP8 stores weight / scale: its largest value touches the format's
    range (448 in E4M3), and the scale is amax / 448. The ratio
    max|stored| / (scale x range) is therefore 1 / scale, that is 71 to 1,691 across the 16
    FP8/INT8 files of the library. kleinFinalcutFP16FP8_comfyQuant stores the weights AS IS
    and still supplies amax / 448: ratio 1.03. Applying that scale made every weight 1,200
    to 1,700 times too small, the transformer produced nothing any more and the image came
    out as noise, for any prompt.
    MX scales (uint8 = an E8M0 exponent) mean something else: never concerned.
"""
    rng = _QUANT_RANGE.get(qdtype)
    fmt = str((cfg or {}).get("format", "")).lower()
    if rng is None or s.dtype == torch.uint8 or fmt.startswith("mx"):
        return False
    smax = float(s.detach().float().abs().max())
    if smax <= 0.0:
        return False
    amax = float(t.detach().float().abs().max())
    # 1. A normal 'scaled' file FILLS the format's range (weight / scale touches 448
    #    or 127); a file already at scale leaves it nearly empty (0.375 out of 448 for
    #    comfyQuant). Without this condition, a full INT8 (+-127) whose scale is near 1
    #    passed for 'already at scale' (seen on test_int8_per_row_scale).
    if amax >= rng / 4:
        return False
    # 2. ... AND the scale describes exactly the stored values: ratio ~1. Well below
    #    that (an arbitrary scale), it is not this case.
    ratio = amax / (smax * rng)
    return 0.5 <= ratio < 2.0


_PRESCALED = {}      # file key -> bool (read once per file and per session)


def _source_prescaled(src):
    """Does the source file store its weights already at scale? Read on the SMALLEST
    quantized tensor that carries a scale: a few KB to read, even on the USB disk.
    Any error = False: the cache key then does not change."""
    try:
        fk = _file_key(src)
    except OSError:
        return False
    if fk in _PRESCALED:
        return _PRESCALED[fk]
    res = False
    try:
        hdr = _safetensors_header(src)
        best = None
        for k, v in hdr.items():
            if not (isinstance(v, dict) and k.endswith(".weight")):
                continue
            if str(v.get("dtype", "")).upper() not in ("F8_E4M3", "F8_E5M2", "I8"):
                continue
            sk = next((c for c in (k + "_scale", k[:-len(".weight")] + ".scale_weight")
                       if c in hdr), None)
            if sk is None:
                continue
            n = 1
            for d in v.get("shape") or [1]:
                n *= int(d)
            if best is None or n < best[0]:
                best = (n, k, sk)
        if best:
            from safetensors import safe_open
            with safe_open(src, framework="pt", device="cpu") as f:
                t = f.get_tensor(best[1])
                s = f.get_tensor(best[2])
            res = _stored_at_scale(t, s, t.dtype)
    except Exception as e:
        _dbg(f"prescaled check failed on {os.path.basename(src)}: {e}")
        res = False
    _PRESCALED[fk] = res
    return res


def _apply_quant_scale(t, s, key, path, cfg=None):
    """Applies the dequantization scale, whatever its granularity.

    Three shapes exist in the wild, and the third one crashed the load deep inside
    torch on "The size of tensor a (4096) must match the size of tensor b (128)",
    naming neither the file nor the format:
      - scalar / [1]          -> one scale for the whole tensor
      - [out] / [out, 1]      -> one scale per output row
      - [out, nb]             -> PER BLOCK: nb groups along the input, each covering
                                 in/nb elements (seen at 32 on a klein-9B FP8)
    Any other shape is refused WITH its dimensions: an unknown format must say so,
    not be guessed.
"""
    # MXFP8 (OCP microscaling, what ComfyUI produces on FLUX.2): the scale is a uint8
    # encoding an E8M0 EXPONENT, not a multiplier. Reading it as a linear factor gives
    # weights ~10000x too large -- the image comes out as mush or as NaN, without any step
    # complaining. The file declares its format in the `comfy_quant` blob; failing that, a
    # uint8 can only be an exponent.
    fmt = str((cfg or {}).get("format", "")).lower()
    if fmt.startswith("mx") or s.dtype == torch.uint8:
        s = torch.exp2(s.to(torch.float32) - 127.0)
    else:
        s = s.to(torch.float32)
    if s.dim() == 0 or s.numel() == 1:
        return t * s.reshape(())
    if s.dim() == 1 and t.dim() == 2 and s.shape[0] == t.shape[0]:
        return t * s.unsqueeze(1)
    if s.dim() == 2 and s.shape[0] == t.shape[0]:
        if s.shape[1] == 1:
            return t * s
        nb = s.shape[1]
        if t.dim() == 2 and t.shape[1] % nb == 0:
            g = t.shape[1] // nb          # group size along the input
            return (t.view(t.shape[0], nb, g) * s.unsqueeze(-1)).view(t.shape[0], -1)
    raise RuntimeError(
        f"{os.path.basename(path)}: unsupported FP8/INT8 scale layout on '{key}' "
        f"(weight {tuple(t.shape)}, scale {tuple(s.shape)}). Known layouts: one "
        f"scale for the tensor, one per output row, or one per block along the "
        f"input dimension. Please report the file.")


def _load_dequant_state_dict(path):
    """Loads a ComfyUI single-file into RAM and returns it IN THE DIFFUSERS LAYOUT,
    dequantized to DTYPE (bf16) tensor by tensor. Serves both cases: quantized (FP8/INT8
    'scaled') and merely PREFIXED (bf16/fp16 -- nothing to dequantize, just the prefix to
    strip):
      - an AIO bundle (transformer + text encoder + VAE): only the
        'model.diffusion_model.*' keys are kept (VAE + encoder = the base repo's);
      - X.weight (F8/I8) * X.weight_scale (scalar or per row) -> bf16;
      - the X.comfy_quant blob: when 'convrot' is declared (ComfyUI int8_tensorwise), the
        grouped Hadamard rotation (256 by default) is UNDONE after the descale -- without
        that the weights are pure noise;
      - the quantization keys (weight_scale/scale_weight, comfy_quant, the scaled_fp8
        marker) are consumed and dropped.
    The resulting dict goes into from_single_file (diffusers key conversion included).
    VRAM/RAM note: dequantized = the footprint of a full BF16; FP8 only saves disk and
    download, not memory.
"""
    from safetensors import safe_open
    t0 = time.time()
    hdr = _safetensors_header(path)
    entries = [(k, v) for k, v in hdr.items()
               if k != "__metadata__" and isinstance(v, dict)]
    # AIO bundle: keep only the transformer. (No ComfyUI prefix = a transformer-only
    # file in the original layout -> no filtering.) crispz-krea2's method: the prefix is
    # stripped at READ time, so everything downstream (scales, qcfg, architecture guard,
    # returned state dict) works on diffusers-layout keys, with no variant.
    prefix = ""
    if any(k.startswith(_COMFY_PREFIX) for k, _ in entries):
        prefix = _COMFY_PREFIX
        entries = [(k, v) for k, v in entries if k.startswith(prefix)]
    # Architecture guard: a quantized checkpoint of ANOTHER model (keys without a
    # single Qwen-Image marker) would load incoherent weights -> a clear refusal.
    if not any(any(m in k[len(prefix):] for m in _QWEN_KEY_MARKERS) for k, _ in entries):
        raise RuntimeError(
            f"{os.path.basename(path)}: quantized checkpoint does not look like a "
            "FLUX.2 transformer (different architecture); this build only loads "
            "FLUX.2 Klein models.")
    # SEQUENTIAL read in the file's PHYSICAL order (data_offsets): a HDD collapses on
    # random access, and the key order does not follow the data's.
    entries.sort(key=lambda kv: kv[1].get("data_offsets", [0])[0])
    raw = {}
    qcfg = {}
    # comfy-quants declares the scheme either in PER-TENSOR blobs (X.comfy_quant), or
    # CENTRALLY in __metadata__._quantization_metadata (the StableYogi variant:
    # {"layers": {"blocks...": {"format": "int8_tensorwise", "convrot": true,
    # "convrot_groupsize": 256}}}). Ignoring that variant leaves the rotation in place ->
    # weights as pure noise (observed on the Krea 2 INT8s; the same format is possible
    # here). The per-tensor blobs win.
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
    # The dequantization work (fp32 cast + scales + un-rotation) is MEMORY BANDWIDTH
    # bound on the CPU (measured on crispz-krea2: ~9 min on a 12.9B INT8): so it runs on
    # the GPU when there is one, tensor by tensor (a few hundred MB of VRAM at most), with
    # the bf16 coming back to RAM. config convert_device: auto (the default) | cpu.
    dev = "cpu"
    try:
        if (torch.cuda.is_available()
                and str(CONFIG.get("convert_device", "auto")).lower() != "cpu"):
            dev = "cuda"
    except Exception:
        pass
    _had = {}                                # a Hadamard cache per group size
    sd = {}
    n_dq = n_rot = n_pre = 0
    for k in list(raw.keys()):
        if (k.endswith((".weight_scale", ".scale_weight", ".scale_input", ".input_scale"))
                or k.endswith("scaled_fp8")):
            continue                         # consommees via lookup / jetees (scale_input
                                             # = an ACTIVATION scale, not a weight one)
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
            qdt = t.dtype
            t = t.to(dev).to(torch.float32)
            cfg0 = qcfg.get(k[:-len(".weight")]) if k.endswith(".weight") else None
            if s is not None and _stored_at_scale(t, s, qdt, cfg0):
                s = None                     # already at scale: see _stored_at_scale
                n_pre += 1
            if s is not None:
                t = _apply_quant_scale(t, s.to(dev), k, path, cfg0)
            # ConvRot (comfy-quants int8_tensorwise): the stored weights were rotated
            # W_rot = (W.view(out, in/g, g) @ H.T).reshape(...) BEFORE quantization ->
            # reconstruction = multiply by H again (orthonormal, symmetric) per group.
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
         + (f", {n_pre} already stored at scale: weight_scale NOT applied" if n_pre else "")
         + f") to bf16 in {time.time() - t0:.1f}s")
    return sd


# Architectures accepted in .gguf files. A diffusion GGUF declares its architecture in
# 'general.architecture': 'flux'/'flux2' (FLUX.1 as well as FLUX.2 -- the label does NOT
# distinguish them), 'qwen_image', 'krea2', 'llama'/'gemma3' for LLMs.
# So the label is accepted broadly and the LAYOUT decides (_gguf_layout): FLUX.2's tensor
# names are proof, not a declaration.
# Overridable through config 'gguf_arch' (a string, or a comma-separated list).
GGUF_ARCH = str(CONFIG.get("gguf_arch") or "flux2,flux").strip().lower()
GGUF_ARCHS = {a.strip() for a in GGUF_ARCH.split(",") if a.strip()}

_GGUF_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _gguf_skip(f, t):
    """Skips past a GGUF value in the stream without reading it (strings and arrays
    included)."""
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
    """The 'general.architecture' of a .gguf -- reads the header only (a few KB), never
    the weights. Returns 'qwen_image' / 'flux' / 'krea2' / 'llama'... or None when
    unreadable (in that case nothing is filtered: better to try than to discard a valid
    model)."""
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


# Tensor prefixes of the ORIGINAL Qwen layout (the one diffusers' GGUF loader knows how
# to map -- QuantStack/city96 GGUFs). Some Civitai GGUFs are converted with a compact
# renamed scheme (blocks.N.attn.wq, txtmlp, tproj... -- a stable-diffusion.cpp-style tool):
# the declared architecture really is 'qwen_image' but NO key matches -> every weight stays
# on the 'meta' device and the .to(device) blows up with "Cannot copy out of meta tensor".
# That case is detected from the header so it can be refused cleanly.
_GGUF_OK_PREFIXES = ("transformer_blocks.", "single_transformer_blocks.",
                     "x_embedder", "context_embedder", "double_stream_modulation",
                     "single_stream_modulation", "time_guidance_embed",
                     "norm_out", "proj_out")

# The signature that POSITIVELY identifies a FLUX.2 transformer, as opposed to the other
# diffusers DiTs. The prefixes above are too loose: 'transformer_blocks.', 'norm_out' and
# 'proj_out' exist in Qwen-Image and FLUX.1 TOO. These three keys do not -- read off the
# real transformer of FLUX.2-klein-4B (169 tensors).
# The constant's name is kept to limit the conflict surface when merging upstream.
_QWEN_GGUF_SIGNATURE = ("x_embedder.weight", "context_embedder.weight",
                        "double_stream_modulation_img.linear.weight")


def _gguf_layout(path):
    """Layout state of a .gguf's tensors, read from the header (gguf mmap):
      'flux2'   -> the FLUX.2 signature is there: that is PROOF, far more reliable than
                   the declared 'general.architecture' (conversion tools stamp anything --
                   'wan' has been seen on perfectly valid Qwen-Image files);
      'foreign' -> readable names, but no known diffusers marker (a
                   stable-diffusion.cpp-style conversion: blocks.N.attn.wq, txtmlp,
                   tproj...);
      'unknown' -> an unreadable header, or a diffusers layout without the FLUX.2
                   signature: nothing is decided here, the declared architecture stays the
                   judge.
"""
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


def _gguf_hidden_dim(path):
    """Hidden dimension of a FLUX.2 .gguf (the same signature as the single-file one)."""
    try:
        from gguf import GGUFReader
        # shape gguf = ordre inverse de torch -> on remet (out, in)
        return _flux2_hidden_dim_from_shapes(
            (t.name, list(reversed([int(x) for x in t.shape]))) for t in GGUFReader(path).tensors)
    except Exception as e:
        _dbg(f"gguf hidden dim read failed {path}: {e}")
        return None


def _gguf_layout_unsupported(path):
    """Returns a reason (str) when the .gguf is NOT loadable: a tensor layout diffusers
    does not know, or a FLUX.2 variant (4B/9B) that does not match the base repo. Header
    read only."""
    bad = _flux2_variant_mismatch(_gguf_hidden_dim(path))
    if bad:
        return bad
    if _gguf_layout(path) != "foreign":
        return None
    return ("GGUF with a non-standard tensor layout (e.g. stable-diffusion.cpp "
            "conversion); diffusers cannot map it — use a QuantStack/city96-style "
            "GGUF or the BF16/FP16 .safetensors build")


def _checkpoint_dirs():
    """Folders to scan for single-file checkpoints: the main one + the extra one (when
    set), with no duplicate path."""
    dirs = [CHECKPOINTS_DIR]
    if CHECKPOINTS_EXTRA_DIR and CHECKPOINTS_EXTRA_DIR not in dirs:
        dirs.append(CHECKPOINTS_EXTRA_DIR)
    return dirs


def list_checkpoints():
    """Single-file FLUX.2 models (.safetensors / .gguf) from the checkpoints folders (main
    + extra, merged into a single list). ComfyUI 'scaled' FP8/INT8 are accepted (the dequant
    loader, see _safetensors_dequant); only these stay discarded, with their reason: stray
    LoRAs, SVDQuant/Nunchaku INT4, GGUFs of another architecture or in the sd.cpp layout,
    and the builds of the OTHER variant (4B/9B).

    VARIANT mismatches are grouped: a library can hold dozens of klein-9B, and repeating
    the instructions on every line drowns the log. One short line per file, then ONE
    summary that says what to do.

    On a duplicate file name, the main folder wins.
"""
    out = []
    seen = set()
    variant_skips = {}          # dim -> number of discarded files
    for d in _checkpoint_dirs():
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f in seen:
                continue
            if not f.lower().endswith((".safetensors", ".ckpt", ".pt", ".sft", ".gguf")):
                continue
            if f.lower().endswith(".safetensors"):
                fp = os.path.join(d, f)
                dim = _flux2_hidden_dim(fp)
                if _flux2_variant_mismatch(dim):
                    variant_skips[dim] = variant_skips.get(dim, 0) + 1
                    _dbg(f"checkpoint skipped ({_variant_name(dim)}): {f}")
                    continue
                reason = _safetensors_unsupported(fp)
                if reason:
                    _log(f"checkpoint skipped ({reason}): {f}")
                    continue
            if f.lower().endswith(".gguf"):
                fp = os.path.join(d, f)
                dim = _gguf_hidden_dim(fp)
                if _flux2_variant_mismatch(dim):
                    variant_skips[dim] = variant_skips.get(dim, 0) + 1
                    _dbg(f"checkpoint skipped ({_variant_name(dim)}): {f}")
                    continue
                lay, a = _gguf_layout(fp), _gguf_arch(fp)
                # The LAYOUT beats the declared architecture: tensor names are proof,
                # the 'general.architecture' KV a mere label -- and conversion tools stamp
                # it wrong (Qwen-Image files published as 'wan').
                if lay == "foreign":
                    _log(f"checkpoint skipped ({_gguf_layout_unsupported(fp)}): {f}")
                    continue
                if lay == "flux2":
                    if a and a not in GGUF_ARCHS:
                        _log(f"GGUF declares architecture '{a}' but its tensors ARE a "
                             f"FLUX.2 transformer (mislabelled by the conversion "
                             f"tool) -> loaded anyway: {f}")
                # layout undetermined -> the declared architecture stays the judge.
                elif a and a not in GGUF_ARCHS:
                    _log(f"checkpoint skipped (GGUF architecture '{a}', this build only "
                         f"loads {sorted(GGUF_ARCHS)}; that model needs its own pipeline "
                         f"and text encoder/VAE): {f}")
                    continue
            seen.add(f)
            out.append(f)
    for dim, n in sorted(variant_skips.items()):
        _log(_variant_skip_summary(n, dim))
    return sorted(out)


def checkpoint_refusal(name):
    """A reason (str) why THIS checkpoint is not selectable, otherwise None.

    The same verdict as list_checkpoints, but for a single file and with the full
    instructions: this is what someone reads who is trying to pick that model and not
    another (a preset written before a base repo change, a moved checkpoint, a GGUF of
    another architecture). None for an HF repo / diffusers folder: only single-file
    checkpoints go through this filter.
"""
    if not name:
        return None
    path = name if os.path.isabs(name) else resolve_checkpoint(name)
    if not _looks_single_file(path):
        return None
    if not os.path.isfile(path):
        return (f"no such file in the checkpoint folder(s) "
                f"({', '.join(_checkpoint_dirs())})")
    if _is_gguf_path(path):
        dim = _gguf_hidden_dim(path)
        if _flux2_variant_mismatch(dim):
            return _variant_refusal(dim)
        lay, arch = _gguf_layout(path), _gguf_arch(path)
        if lay == "foreign":
            return _gguf_layout_unsupported(path)
        # As in list_checkpoints: the layout beats the declared architecture.
        if lay != "flux2" and arch and arch not in GGUF_ARCHS:
            return (f"its GGUF architecture is '{arch}' and this build only loads "
                    f"{sorted(GGUF_ARCHS)}; that model needs its own pipeline and "
                    f"text encoder/VAE")
        return None
    dim = _flux2_hidden_dim(path)
    if _flux2_variant_mismatch(dim):
        return _variant_refusal(dim)
    return _safetensors_unsupported(path)


def resolve_checkpoint(name):
    """Absolute path of a single-file checkpoint from its file name, looked up in the
    checkpoints folders (main then extra). Returns name as is when it is already absolute;
    falls back to the main folder when not found."""
    if not name or os.path.isabs(name):
        return name
    for d in _checkpoint_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(CHECKPOINTS_DIR, name)


def list_loras():
    """LoRAs (.safetensors / .ckpt / .pt) from the loras folder, RECURSIVELY (subfolders
    included). Returns paths RELATIVE to LORAS_DIR with '/' (e.g.
    'subfolder/my_lora.safetensors') -> set_loras / resolve resolve them through
    os.path.join(LORAS_DIR, name)."""
    exts = (".safetensors", ".ckpt", ".pt")
    out, seen = [], set()
    for d in _lora_dirs():          # main then extras: same name -> the main one wins
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
    """Sets (or clears with '' / None) the additional checkpoints folder."""
    global CHECKPOINTS_EXTRA_DIR
    CHECKPOINTS_EXTRA_DIR = (path or "").strip()


def set_loras_dir(path):
    global LORAS_DIR
    if path:
        LORAS_DIR = path


def set_loras_extra_dirs(spec):
    """Sets (or clears with '' / [] / None) the extra LoRA folders.
    spec = a list or an 'a;b' string."""
    global LORAS_EXTRA_DIRS
    LORAS_EXTRA_DIRS = _split_dirs(spec)


def _read_safetensors_metadata(path):
    """Reads the JSON header (__metadata__) of a .safetensors WITHOUT loading the weights."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
    return (json.loads(header.decode("utf-8")) or {}).get("__metadata__", {}) or {}


def lora_keywords(path):
    """Extracts a LoRA's keywords / trigger words from its metadata: explicit trigger
    fields + the top training tags (ss_tag_frequency)."""
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
    """Sets the active LoRAs. slots = a list of (name_or_None, weight). Resolves the
    names to paths, ignores the Nones.

    Does NOT reload the model: the LoRAs are hot-swapped on the transformer already in
    VRAM (_apply_loras, called by _ensure_base on the next run).
"""
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
    """Sets the EDIT LoRAs (the omni pipe). slots = a list of (name_or_None, weight); a
    name is a cz_edit_loras preset (downloaded on demand), an absolute path, or a file
    relative to LORAS_DIR. Ignores the Nones. Applied hot on the next generate_omni
    (_apply_edit_loras). Raises when a preset cannot be downloaded."""
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
    """Fast edit mode. 'Off'/None -> steps/guidance from Settings, no speed LoRA.
    'Auto' -> the edit model's model_profiles profile (Rapid-AIO / a Lightning merge:
    already distilled, 4-8 steps, CFG off), with no LoRA. 'Lightning N steps' -> the
    Lightning LoRA (2509 or 2511 depending on the edit model; looked up in the LoRA
    folders, downloaded otherwise) + N steps + guidance 1.0. Returns the applied dict."""
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
    """The 'Edit LoRAs' checkbox: ON = the EDIT_LORAS set is applied on the omni pipe,
    OFF = removed (the set stays remembered). No reload."""
    global EDIT_LORAS_ENABLED
    on = bool(on)
    if on != EDIT_LORAS_ENABLED:
        EDIT_LORAS_ENABLED = on
        _log(f"edit LoRAs {'enabled' if on else 'disabled'}")


def set_omni_model(repo):
    """A no-op on klein: multi-reference editing is served by the BASE pipeline, there is
    no separate edit model to choose. The function survives because cz_ui wires it to the
    'Omni model' dropdown (upstream's API contract); all it does is point OMNI_MODEL back at
    BASE_REPO and log it.
    To change the edit model on klein, you change the model itself (set_zimage_model /
    the Checkpoint dropdown).
"""
    global OMNI_MODEL
    OMNI_MODEL = BASE_REPO
    if (repo or "").strip() and (repo or "").strip() != BASE_REPO:
        _log(f"Omni model ignored ({repo}): klein edits with the base model "
             f"({BASE_REPO}). Change the checkpoint to change the editor.")


def list_edit_models():
    """Available EDIT models. On klein the editor IS the base model, so any loadable klein
    checkpoint will do: the same list as the checkpoints is returned (cz_ui's 'Omni model'
    dropdown stays fed and coherent)."""
    return list_checkpoints()


def check_omni_available():
    """On klein, multi-reference editing is NATIVE to the base pipeline: it is available as
    soon as the model is loadable, with no second download and no repo to check on the Hub.
    So a ready message is always returned (cz_ui's contract wants a markdown string)."""
    return (f"**Edit ready (native):** `{BASE_REPO}` handles multi-reference editing in "
            f"the SAME pipeline as txt2img (up to 4 refs) - no second model, no extra "
            f"VRAM. Note: klein is distilled, so **negative prompts and CFG have no "
            f"effect** (see FORK.md).")


@_gpu_exclusive
def set_offload_mode(mode):
    """Changes the CPU offload mode. Invalidates the pipe (the hooks are set at load
    time). An unknown value -> 'auto' (never 'none': the fallback must be the SAFE mode)."""
    global OFFLOAD_MODE, _AUTO_OFFLOAD
    mode = str(mode or "").strip().lower()
    mode = mode if mode in OFFLOAD_CHOICES else "auto"
    if mode != OFFLOAD_MODE:
        OFFLOAD_MODE = mode
        _AUTO_OFFLOAD = ""   # 'auto' runs the VRAM test again on the next load
        free_vram()
        _log(f"offload -> {OFFLOAD_MODE}: pipeline invalidated -> will reload")


# ---- Offload 'auto': a VRAM test at load time + a runtime safety net (cz_hw) ----

def _hw_profile_path():
    """Profile of the VRAM test's verdicts (JSON), next to the other caches."""
    return os.path.join(HERE, "cache", "hw_profile.json")


def _model_footprint_gb():
    """VRAM footprint (GB) of the whole pipeline under 'none' offload (weights in VRAM,
    activations excluded). Built on the per-variant estimate _base_vram_need_gb
    (klein-4B ~15 GB, 9B ~35 GB, encoder pruning included), adjusted so that the TOTAL
    margin stays _VRAM_HEADROOM_GB (cz_hw already adds 2.5 GB of activations at
    1024x1024). Overridable through config 'model_footprint_gb'."""
    try:
        v = float(CONFIG.get("model_footprint_gb", 0) or 0)
        if v > 0:
            return v
    except Exception:
        pass
    try:
        need = _base_vram_need_gb()
        if need:
            return float(need) + max(0.0, _VRAM_HEADROOM_GB - 2.5)
    except Exception:
        pass
    return 15.0


def _resolve_auto(retest=False):
    """The concrete mode for 'auto' (memoised for the process). The verdict is cached in
    cache/hw_profile.json per (GPU, torch/cuda build, model, dtype): the test only costs one
    mem_get_info per combination, then a JSON read."""
    global _AUTO_OFFLOAD
    if _AUTO_OFFLOAD and not retest:
        return _AUTO_OFFLOAD
    mode, why = cz_hw.resolve(
        "auto", footprint_gb=_model_footprint_gb(),
        model_id=(ZIMAGE_TRANSFORMER or BASE_REPO), dtype="bf16",
        profile_path=_hw_profile_path(), retest=retest)
    _AUTO_OFFLOAD = mode
    _log(f"offload auto -> {mode} ({why})")
    return mode


def offload_status():
    """Status line for the UI: the requested mode + the 'auto' resolution when relevant."""
    if OFFLOAD_MODE != "auto":
        return f"offload: {OFFLOAD_MODE} (explicit)"
    if not _AUTO_OFFLOAD:
        return "offload: auto (resolves at the next model load)"
    return f"offload: auto -> {_AUTO_OFFLOAD}"


def retest_offload():
    """The UI's 'Re-test VRAM' button: runs the test again, ignoring the profile (another
    app closed/opened, a driver change...). Invalidates the pipe when the verdict changes."""
    if OFFLOAD_MODE != "auto":
        return f"Offload is '{OFFLOAD_MODE}' (explicit) - select 'auto' to use the VRAM test."
    old = _AUTO_OFFLOAD
    mode = _resolve_auto(retest=True)
    if old and mode != old:
        free_vram()
        return f"auto -> {mode} (was {old}; the pipeline will reload)"
    return f"auto -> {mode}"


def _vram_guard_kwargs():
    """Runtime safety net: a callback_on_step_end that checks AFTER the first denoise step
    in effective mode 'none' that the VRAM is not saturated (the load-time test estimates; a
    third-party process may have arrived since, or the requested resolution exceeds the
    margin). Saturated -> the flag + an interruption of the denoise; the caller switches to
    'model' and replays the job ONCE.
    {} when the guard is pointless (offload already on, no CUDA).
"""
    if DEVICE != "cuda" or _effective_offload() != "none":
        return {}

    def _cb(pipe, i, t, cb_kwargs):
        global _VRAM_DOWNGRADE
        if i == 0 and cz_hw.vram_saturated():
            _VRAM_DOWNGRADE = True
            pipe._interrupt = True
        return cb_kwargs
    return {"callback_on_step_end": _cb}


def _consume_vram_downgrade():
    """When the guard has fired: applies the downgrade to 'model', records it in the
    profile (the next boot starts in 'model' directly) and releases the pipe. True -> the
    caller replays the job once."""
    global _VRAM_DOWNGRADE, _AUTO_OFFLOAD
    if not _VRAM_DOWNGRADE:
        return False
    _VRAM_DOWNGRADE = False
    _log("WARNING: VRAM saturated after the first denoise step in offload 'none' "
         "-> the render would spill to shared RAM (50-100x slower, no error). "
         "Switching to 'model' and retrying the job once.")
    cz_hw.record_downgrade(_hw_profile_path(), ZIMAGE_TRANSFORMER or BASE_REPO,
                           "bf16", "model", "VRAM saturated after denoise step 1")
    if OFFLOAD_MODE == "auto":
        _AUTO_OFFLOAD = "model"
        free_vram()
    else:
        set_offload_mode("model")
    return True


@_gpu_exclusive
def free_vram():
    """Releases the base pipeline + the derived pipelines and gives the VRAM back
    (step 3: unload on idle or the /unload endpoint). Lazy reload."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _APPLIED_LORAS, _APPLIED_LOKRS
    global _ENCODER_TRIMMED, _TEXT_ENCODER_ACTIVE
    _BASE_PIPE = None
    _DERIVED = {}
    _LOADED_KEY = None
    _APPLIED_LORAS = []      # no pipe any more -> no adapter applied either
    _APPLIED_LOKRS = []      # ... nor of weights where a LoKr would be merged
    _ENCODER_TRIMMED = False # ... nor of a pruned encoder
    _TEXT_ENCODER_ACTIVE = ""  # ... nor of a replacement encoder loaded
    _embed_cache_clear(" (VRAM freed)")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def is_oom(e):
    """True when `e` is a lack of VRAM, in either of its two forms: torch's allocator one
    ("CUDA out of memory. Tried to allocate ...") and a direct CUDA call's one ("CUDA error:
    out of memory"). The second happens once torch's cache has reserved everything: a kernel
    loaded on demand, or onnxruntime (the detailer's face detector), finds nothing left and
    cannot claim anything back from that cache."""
    s = str(e).lower()
    return "out of memory" in s or "alloc_failed" in s


def release_vram(offload=False, why=""):
    """Gives the driver back the VRAM that torch's cache holds in reserve, unloading
    nothing.

    torch only empties its cache when ITS allocator fails; the other consumers fail without
    being able to reclaim it. offload=True also puts back on the CPU the models an
    interrupted call left on the GPU under 'model' offload (a transformer half-moved by an
    OOM stayed there: 10.8 GB stuck, and every later render failed until a restart). `why`
    logs the VRAM state afterwards.
"""
    if offload and _BASE_PIPE is not None and getattr(_BASE_PIPE, "_all_hooks", None):
        try:
            _BASE_PIPE.maybe_free_model_hooks()   # diffusers: everything on the CPU, hooks put back
        except Exception as e:
            _dbg(f"release_vram: offload failed ({e})")
    gc.collect()
    if DEVICE != "cuda":
        return
    try:
        torch.cuda.empty_cache()
        if why:
            free, total = torch.cuda.mem_get_info()
            _log(f"VRAM released ({why}): {free / 1024 ** 3:.1f} GB free of "
                 f"{total / 1024 ** 3:.1f}, torch holds "
                 f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB "
                 f"(reserved {torch.cuda.memory_reserved() / 1024 ** 3:.1f})")
    except Exception as e:
        _dbg(f"release_vram: {e}")


def _load_lora(pipe, *args, **kwargs):
    """pipe.load_lora_weights with REAL tensors (low_cpu_mem_usage=False).

    A diffusers/peft build that does not know that parameter refuses it with a TypeError:
    the call is then retried without it rather than failing the application (the default
    creates the layers on 'meta' there -- see _sync_adapters).
"""
    try:
        return pipe.load_lora_weights(*args, low_cpu_mem_usage=False, **kwargs)
    except TypeError as e:
        if "low_cpu_mem_usage" not in str(e):
            raise
        _dbg(f"load_lora_weights without low_cpu_mem_usage ({e})")
        return pipe.load_lora_weights(*args, **kwargs)


def _offload_hooks(pipe):
    """Number of 'model' offload hooks diffusers has set on this pipe (0 = none)."""
    return len(getattr(pipe, "_all_hooks", None) or [])


def restore_offload(pipe, why=""):
    """Puts the pipe back in its EFFECTIVE offload state when it has been left on the CPU.

    diffusers REMOVES the offload hooks before applying a LoRA and puts them back after.
    When the load fails in between, nobody puts them back: the pipe stays on the CPU, its
    `_execution_device` becomes cpu, and EVERY later render fails on "Cannot generate a cpu
    tensor from a generator of type cuda" -- until the app is restarted. Caught on
    2026-09-23 with two DoRA LoRAs applied in editing. Returns True when the state has been
    restored.
"""
    if DEVICE != "cuda" or pipe is None:
        return False
    try:
        dev = pipe._execution_device
    except Exception:
        return False
    if str(getattr(dev, "type", dev)) == "cuda":
        return False
    off = _effective_offload()
    try:
        if off == "model":
            pipe.enable_model_cpu_offload()
        elif off == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(DEVICE)
    except Exception as e:
        _log(f"pipeline left on the CPU and NOT restored ({e}): restart crispz-klein")
        return False
    _log(f"pipeline was left on the CPU{' after ' + why if why else ''} -> offload "
         f"'{off}' restored in place (no reload)")
    return True


def retry_on_oom(what, fn, *args, **kwargs):
    """Calls fn(*args, **kwargs); on a lack of VRAM, gives the VRAM back (torch's cache,
    models left on the GPU) and retries ONCE. A second failure gives the VRAM back again
    before re-raising: the process stays usable for the next render."""
    err = None
    for attempt in (1, 2):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not is_oom(e):
                raise
            # The traceback holds the frames, so their tensors on the GPU: it is
            # dropped BEFORE emptying the cache, or empty_cache reclaims nothing.
            err = e.with_traceback(None)
            err.__context__ = err.__cause__ = None
        if attempt == 1:
            _log(f"{what}: out of VRAM ({str(err).strip().splitlines()[0]}), "
                 f"freeing it and retrying once")
        release_vram(offload=True, why=what)
    raise err


# Beyond this side (px) attention slicing is turned on (whole-image 2K+ -> avoids the
# 32 GB VRAM spill). Below it (1024 tiles, 1024/1536 txt2img) -> slicing OFF = native SDPA
# attention = FAST (like ComfyUI). Tunable through config attention_slice_above.
_SLICE_ABOVE = int(CONFIG.get("attention_slice_above", 1664))

# Guard rail: beyond this side (px), a "whole image" refine (refine_tile=0) is
# auto-tiled (1024 tile). The default = the slicing threshold: beyond it a whole-image pass
# would be sliced (slow: ~120s at 2K) AND risks the VRAM spill (4K -> a crash). Tiling is
# faster AND safe.
_AUTO_TILE_ABOVE = int(CONFIG.get("auto_refine_tile_above", _SLICE_ABOVE))

# Size of the tile this auto-tiling uses. "auto" (the default) = computed by
# _pick_refine_tile; an integer freezes the size (the old behaviour: 1024).
# Measured (RTX 5090, 4096x4096 output, denoise 0.40, overlap 64): the cost per pixel is
# FLAT from 768 to 1024 (1.78 / 1.83 / 1.79 us/px) and only climbs beyond that (2.41 at
# 1536, 3.00 at 2048). So the time follows the TILED AREA (n x tile^2), not the tile size.
# But at 1024 the grid overflows: 960 does not divide 4096 -> the last tile is pulled back
# and overlaps the previous one by 832px instead of 64, that is 1.56x the image's area. At
# 896 the step lands right (1.20x) -> 36.7s instead of 46.9s on the same image, with an
# IDENTICAL number of tiles (25) and seams (8).
# Bounds [768, 1024]: below them tiles and seams multiply and each tile sees less context
# (the render drifts - a blurred background rebuilds differently, checked visually); above
# them the attention becomes superlinear.
_AUTO_TILE_MIN = int(CONFIG.get("auto_refine_tile_min", 768))
_AUTO_TILE_MAX = int(CONFIG.get("auto_refine_tile_max", 1024))
_AUTO_TILE_SIZE = str(CONFIG.get("auto_refine_tile", "auto")).strip().lower()


def _pick_refine_tile(w, h, overlap):
    """The tile that minimises the tiled area needed to cover w x h (= the pass's real
    cost).

    At equal area the LARGEST tile wins: fewer seams and more context per tile. An integer
    in auto_refine_tile short-circuits the computation (a frozen size).
"""
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
        cands.append((n * t * t, -t, t))       # the smallest area, then the largest tile
    return min(cands)[2]

# Denoise ceiling for the TILED refine. In tiles, each tile is re-diffused with the
# global prompt -> at a high denoise the diffusion rebuilds the subject (the cup, say) IN
# every tile = duplications. So the per-tile denoise is capped (the existing content then
# guides the diffusion, Ultimate SD Upscale style). The "whole image" refine keeps the
# requested denoise (no duplication is possible: a single pass over the whole composition).
# Tunable through config refine_tile_denoise_cap (0 = no cap).
_TILE_DENOISE_CAP = float(CONFIG.get("refine_tile_denoise_cap", 0.40))

# Prompt used for the TILED refine. The global prompt describes the WHOLE composition
# (not the tile) -> handing it to every tile pushes the diffusion to recreate the subject
# (the cup) in tiles that are nothing but background. So an EMPTY prompt is passed by
# default: each tile just refines the local detail. config refine_tile_prompt values:
#   "" (the default) = an empty prompt per tile
#   "global"/"scene" = reuses the scene's prompt (the old behaviour)
#   any other text = a generic prompt applied to every tile (e.g. "high detail, sharp")
_TILE_PROMPT = str(CONFIG.get("refine_tile_prompt", ""))


def _tile_prompt(scene_prompt):
    """The prompt to use per tile according to the config (empty by default,
    anti-duplication)."""
    if _TILE_PROMPT.strip().lower() in ("global", "scene"):
        return scene_prompt or ""
    return _TILE_PROMPT


def _set_slicing(pipe, longest_side):
    """Turns attention slicing on/off according to the largest side to process. Called
    before EVERY diffusion pass (txt2img/refine/tile/inpaint/outpaint/omni)."""
    try:
        if int(longest_side) > _SLICE_ABOVE:
            pipe.enable_attention_slicing()
        else:
            pipe.disable_attention_slicing()
    except Exception:
        pass


def _vram_str():
    """PyTorch's peak reserved VRAM / the total (to spot saturation -> a spill into
    Windows' shared RAM = extreme slowness, and TDR/'CUDA unknown error'). Does NOT see the
    other processes' VRAM (ComfyUI, etc.) -> use nvidia-smi for the real total."""
    if DEVICE != "cuda":
        return ""
    try:
        resv = torch.cuda.memory_reserved() / 1024**3
        tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f" | VRAM {resv:.1f}/{tot:.0f} Go"
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# Qwen-Image (diffusers, BF16): a "base" txt2img pipeline that owns the components, with
# img2img / inpaint derived through from_pipe (shared weights, no duplicate VRAM).
# ----------------------------------------------------------------------------
def _is_gguf_path(p):
    return bool(p) and str(p).lower().endswith(".gguf")


# VRAM a base repo asks for when placed ENTIRELY on the GPU (offload 'none'), per
# variant: transformer + Qwen3 text encoder + VAE, in bf16. Measured on the published
# weights. Used to refuse a configuration that does not fit BEFORE attempting it.
_BASE_VRAM_GB = {"4B": 15.6, "9B": 33.7}
# What _trim_text_encoder removes (never-read blocks + lm_head). Measured on the real
# weights, not estimated. Counted in the budget ONLY when the pruning is active: otherwise
# we would announce room we do not free.
_ENCODER_TRIM_GB = {"4B": 2.2, "9B": 4.0}
# The TRANSFORMER alone, in bf16 (the rest = the Qwen3 encoder + the VAE). Used to
# correct the estimate when a single-file checkpoint replaces the repo's.
_TRANSFORMER_VRAM_GB = {"4B": 7.2, "9B": 18.2}


def _base_vram_need_gb(base=None):
    """VRAM asked for under 'none' offload by what will REALLY be resident, or None when
    the variant is unknown (in which case we stay out of it).

    A single-file override does not make the model smaller, except as a GGUF: an FP8/INT8
    .safetensors is DEQUANTIZED to bf16 at load time and weighs as much as the original
    transformer (16.9 GB on disk -> 18.2 GB in VRAM, measured on a klein-9B). Counting the
    file, or worse skipping the check as the first version of this guard did, let through a
    configuration that does not fit -- and the failure comes at the first diffusion step,
    after five minutes of dequantization, on a mute "CUDA error: unknown error".
"""
    v = _FLUX2_VARIANTS.get(_base_hidden_dim(base))
    total = _BASE_VRAM_GB.get(v)
    if not total:
        return None
    if _ENCODER_TRIMMED:
        total -= _ENCODER_TRIM_GB.get(v, 0.0)
    t = ZIMAGE_TRANSFORMER
    if not t:
        return total
    rest = total - _TRANSFORMER_VRAM_GB.get(v, 0.0)      # text encoder + VAE
    if _is_gguf_path(t):                                  # stays quantised in VRAM
        try:
            return rest + os.path.getsize(t) / 1024 ** 3
        except OSError:
            return total
    return rest + _TRANSFORMER_VRAM_GB.get(v, 0.0)        # bf16, dequantised or not


def _total_vram_gb():
    try:
        return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Text encoder pruning. FLUX.2 does NOT read the LLM's output: it stacks the hidden states
# of three INTERMEDIATE layers (hence a context_embedder 3 x hidden wide). The layers past
# the last one read, and the lm_head, are therefore computed for every image and then
# thrown away -- on the 9B that is eight blocks of a Qwen3-8B and a projection over 152,000
# tokens.
#
# In a causal transformer, hidden_states[k] is the output AFTER k blocks: the later blocks
# cannot influence it. The pruning is therefore exact, not approximate. Checked bit for bit
# on the 9B (torch.equal on the three layers read), not assumed:
#   15.3 GB -> 11.2 GB, encoding 5.0 s -> 3.2 s, IDENTICAL outputs.
#   4B: 8.2 GB -> 6.0 GB.
#
# The layer indices are READ from diffusers' signature, never hardcoded: if an upstream
# version changed them (9, 18, 27), a frozen constant would produce wrong embeddings
# SILENTLY -- the worst possible failure mode here. Unreadable = no pruning, and we say so.
# ----------------------------------------------------------------------------
TRIM_TEXT_ENCODER = bool(CONFIG.get("trim_text_encoder", True))
# Did the pruning REALLY happen on the current encoder? The VRAM budget only
# subtracts its gain when this flag is true -- never on the INTENTION alone.
# With a measurement behind it: with the wrong method name, the pruning was skipped
# (announced), but the budget still subtracted 4 GB. Result: 29.7 GB announced, 32.3 GB
# really resident, VRAM down to 0.0 GB free. That is exactly the failure mode the offload
# guard exists to prevent, reintroduced through the back door.
_ENCODER_TRIMMED = False


def _encoder_layers_used(pipe):
    """The largest hidden-state index the pipeline really reads, or None when upstream's
    signature does not say.

    The method carrying `hidden_states_layers` is SEARCHED for instead of named: it is
    called `_get_qwen3_prompt_embeds` here, `_get_qwen_prompt_embeds` elsewhere in the same
    family, and guessing that name has already cost a measurement (the pruning did not
    fire, silently for the VRAM budget).
"""
    try:
        import inspect
        for name in dir(type(pipe)):
            if not name.startswith("_get_"):
                continue
            fn = getattr(type(pipe), name, None)
            if not callable(fn):
                continue
            try:
                p = inspect.signature(fn).parameters.get("hidden_states_layers")
            except (TypeError, ValueError):
                continue
            if p is None or p.default is inspect.Parameter.empty:
                continue
            layers = [int(x) for x in (p.default or ())]
            if layers:
                _dbg(f"hidden states read by {name}: {sorted(layers)}")
                return max(layers)
    except Exception as e:
        _dbg(f"hidden_states_layers unreadable: {e}")
    return None


def _trim_text_encoder(pipe):
    """Removes from the encoder the blocks that are never read and the lm_head. No effect
    on the embeddings (proven), to be called BEFORE any move/offload."""
    global _ENCODER_TRIMMED
    _ENCODER_TRIMMED = False        # a new encoder: nothing pruned until we have acted
    if not TRIM_TEXT_ENCODER:
        return
    enc = getattr(pipe, "text_encoder", None)
    layers = getattr(getattr(enc, "model", None), "layers", None)
    if enc is None or layers is None:
        return
    last = _encoder_layers_used(pipe)
    if last is None:
        _log("text encoder NOT trimmed: this diffusers build does not expose which "
             "hidden layers the pipeline reads; keeping every layer is the only safe "
             "answer (set `trim_text_encoder` to false to silence this).")
        return
    keep = last + 1                      # hidden_states[k] = block k's output
    if len(layers) <= keep:
        return
    before = sum(p.numel() for p in enc.parameters()) * 2 / 1024 ** 3
    dropped = len(layers) - keep
    try:
        enc.model.layers = torch.nn.ModuleList(list(layers)[:keep])
        if not isinstance(getattr(enc, "lm_head", None), torch.nn.Identity):
            enc.lm_head = torch.nn.Identity()
    except Exception as e:
        _log(f"text encoder NOT trimmed ({e}); nothing lost, it just stays whole")
        return
    after = sum(p.numel() for p in enc.parameters()) * 2 / 1024 ** 3
    _ENCODER_TRIMMED = True
    _log(f"text encoder trimmed: {dropped} unread block(s) + lm_head dropped "
         f"({len(layers)} -> {keep} layers, {before:.1f} -> {after:.1f} GB). The "
         f"pipeline only reads hidden states up to layer {last}, so the embeddings "
         f"are bit-identical.")


# VRAM margin reserved for what is NOT a weight: the CUDA context, the diffusion's
# activations, the VAE decode. Absolute rather than proportional -- that cost does not
# depend on the card's size, whereas a percentage tightens precisely on the small ones.
#
# The old 0.94 left 1.9 GB on a 32 GB card. But pruning the encoder brings the 9B down to
# 29.7 GB, so UNDER that threshold: it would have gone to 'none' with 0.2 GB of announced
# margin. And under Windows that does not crash -- it SPILLS into shared memory (measured:
# 32.3 GB resident on a 31.8 GB card, 0.0 GB free, no exception), after which rendering
# collapses without the slightest message. Until the real cost of the activations is
# measured, we stay generous. Tunable: `vram_headroom_gb`.
try:
    _VRAM_HEADROOM_GB = float(CONFIG.get("vram_headroom_gb", 4.0))
except (TypeError, ValueError):
    _VRAM_HEADROOM_GB = 4.0


def _effective_offload(tpath=None):
    """The offload REALLY applied.

    Two outright corrections, each because the requested setting can NOT work -- and
    because discovering the failure costs minutes of loading:
      - a quantized GGUF transformer does not move onto the GPU through .to(cuda) nor in
        sequential; only enable_model_cpu_offload puts it on the GPU during the forward;
      - a base repo that does not FIT in VRAM under 'none'. klein-9B asks for ~35 GB
        (transformer 18.2 + Qwen3 8B encoder 16.4): on a 32 GB card it loaded, then died at
        the first diffusion step on a 'CUDA error: unknown error' that does not even name
        the VRAM.
    Both are logged by the caller (_ensure_base).
"""
    off = OFFLOAD_MODE
    if off == "auto":
        off = _resolve_auto()   # test VRAM LIBRE (memoise + profil cache) -> mode concret
    t = ZIMAGE_TRANSFORMER if tpath is None else tpath
    if DEVICE != "cuda":
        return off
    if _is_gguf_path(t) and off != "model":
        return "model"
    if off == "none":
        need, have = _base_vram_need_gb(), _total_vram_gb()
        if need and have and need + _VRAM_HEADROOM_GB > have:
            return "model"
    return off


def _load_transformer(path=None, base=None):
    """Loads ONLY the current transformer (without the rest of the pipeline):
      - a quantized GGUF -> from_single_file + GGUFQuantizationConfig (architecture = the
        base repo's)
      - a single-file .safetensors -> from_single_file
      - an HF repo / diffusers folder -> the 'transformer' subfolder
      - no override -> the base repo's transformer
    Used both on a full load AND for the hot swap (_swap_transformer).
    path/base: by default the BASE's transformer (ZIMAGE_TRANSFORMER / BASE_REPO); the EDIT
    pipe passes its own file (GGUF, FP8 Rapid-AIO...) + its edit repo (zimage_omni_base) for
    the architecture.
"""
    from diffusers import Flux2Transformer2DModel
    path = ZIMAGE_TRANSFORMER if path is None else path
    base = BASE_REPO if base is None else base
    if path:
        if _is_single_file(path):
            # Guard: a file that cannot be loaded (a stray LoRA, an FP8, a quantized
            # one) must fail with an actionable message, not go looking for a default
            # config on the Hub. (No effect on .gguf files: an unreadable header -> None.)
            bad = _safetensors_unsupported(path)
            if bad:
                raise RuntimeError(f"{os.path.basename(path)}: {bad}.")
            if _is_gguf_path(path):
                # A Qwen GGUF transformer (quantized) -> fits in VRAM (~11 GB in Q4)
                # and stays fast. The VAE + text encoder come from the base repo (cached).
                lay = _gguf_layout_unsupported(path)
                if lay:
                    raise RuntimeError(
                        f"{os.path.basename(path)}: {lay}.")
                from diffusers import GGUFQuantizationConfig
                _log(f"loading Klein transformer (GGUF, quantized): {path} ...")
                # config/subfolder = the transformer's architecture from the base repo
                # (cached), otherwise from_single_file does not know the structure and
                # tries a default repo.
                return _load_monitor(
                    f"transformer {os.path.basename(path)} (GGUF)",
                    lambda: Flux2Transformer2DModel.from_single_file(
                        path,
                        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
                        config=base, subfolder="transformer",
                        torch_dtype=DTYPE))
            dq = _safetensors_dequant(path)
            if dq:
                # ComfyUI 'scaled' FP8/INT8 (light Civitai builds) -> dequantized in
                # RAM then the dict is loaded (diffusers key conversion included).
                # Already dequantized once? -> re-read the bf16 from the disk cache: a
                # normal single-file load (seconds) instead of converting the whole file
                # again (minutes on a HDD).
                cached = _dequant_cache_path(path)
                if cached and os.path.isfile(cached):
                    _log(f"loading Klein transformer ({dq} -> bf16, from dequant "
                         f"cache): {os.path.basename(cached)}")
                    try:
                        os.utime(cached, None)       # marks the use for the LRU
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
            # A single-file Qwen checkpoint (.safetensors bf16/fp16) -> a transformer
            # override. config/subfolder = the transformer's architecture from the base repo
            # (already cached), as for the GGUF: without it, from_single_file does not know
            # the structure and goes looking for a default repo -> a failure in offline mode
            # (HF_HUB_OFFLINE=1).
            if _safetensors_comfy_prefixed(path):
                # ComfyUI layout WITHOUT quantization: diffusers does not convert the
                # Qwen keys (identity mapping), and handing the path over directly would
                # leave the model on 'meta'. So we read and strip the prefix ourselves --
                # the same RAM cost, since from_single_file loads the whole checkpoint
                # anyway. No disk cache here: there is nothing dequantized to memorise, it
                # would be a bf16 -> bf16 copy.
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
        # An HF repo / diffusers folder -> load the 'transformer' subfolder.
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
    """Removes EVERY LoRA adapter from the pipe to start from a clean state.

    unload_lora_weights() alone leaves, depending on the diffusers/peft versions, a residual
    peft_config on the transformer -> the next load warns ('Already found a peft_config')
    and, since the same adapter names are reused (cz_lora_i), the old adapter can stay in
    place (the wrong LoRA applied). So the remaining adapters are deleted explicitly by name
    after the unload.
"""
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


# LoRA key dialects. PEFT (what diffusers expects) names the two matrices
# `.lora_A.weight` / `.lora_B.weight`. Other tools write `.lora.down.weight` /
# `.lora.up.weight` (or `.lora_down.` / `.lora_up.`) -- same maths, same rank, down == A and
# up == B. A file that MIXES both (seen on lrzjason/Consistance_Edit_Lora: 160 PEFT keys +
# 40 down/up keys) still loads, but peft only injects what it recognises: the other modules
# get a fresh adapter (B at zero) and the LoRA applies PARTIALLY, without an error.
# So the keys are renamed before loading, and it is logged.
_LORA_ALT_SUFFIXES = ((".lora.down.weight", ".lora_A.weight"),
                      (".lora.up.weight", ".lora_B.weight"),
                      (".lora_down.weight", ".lora_A.weight"),
                      (".lora_up.weight", ".lora_B.weight"))


# kohya naming ("lora_unet_double_blocks_0_...", "lora_te1_..."): diffusers recognises
# it by its `.lora_down.weight` keys and converts it itself, alpha included. Renaming those
# keys before it hid the format from it: we leave it alone.
_KOHYA_PREFIXES = ("lora_unet_", "lora_te")


def _lora_needs_normalizing(path):
    """Does the file contain keys of a non-PEFT dialect, or `.alpha` keys diffusers refuses
    as they are? Header read only. kohya naming is left to diffusers."""
    try:
        h = _safetensors_header(path)
    except Exception:
        return False
    keys = [k for k in h if k != "__metadata__"]
    if any(k.startswith(_KOHYA_PREFIXES) for k in keys):
        return False
    if any(k.endswith(alt) for k in keys for alt, _ in _LORA_ALT_SUFFIXES):
        return True
    has_lora = any(".lora_" in k or ".lora." in k for k in keys)
    return has_lora and any(k.endswith(".alpha") for k in keys)


def _load_lora_normalized(path):
    """State dict of a LoRA with the keys brought back to the PEFT dialect and the
    `.alpha` keys folded into the weights. Returns (state_dict, n_renamed).

    A state dict has no room for a per-module alpha: diffusers refuses the whole file
    ("Make sure all LoRA param names contain 'lora'") and nothing is applied. The alpha is
    an alpha / rank scale on the update (dW = alpha/r * B @ A): it is multiplied into B, as
    diffusers' kohya converter does, then the key is dropped.
    Seen on 2026-09-18 on RealSkin (4B and 9B): diffusers names `transformer.*`, matrices
    `lora_down`/`lora_up`, one `alpha` per module (64, rank 64: scale 1).
"""
    from safetensors.torch import load_file
    sd = load_file(path)
    out, n = {}, 0
    for k, v in sd.items():
        nk = k
        for alt, peft in _LORA_ALT_SUFFIXES:
            if k.endswith(alt):
                nk = k[: -len(alt)] + peft
                n += 1
                break
        out[nk] = v
    folded, scales = 0, set()
    for k in [k for k in out if k.endswith(".alpha")]:
        base = k[: -len(".alpha")]
        a_key, b_key = base + ".lora_A.weight", base + ".lora_B.weight"
        alpha = out.pop(k)          # with no matrices facing it, an alpha has no effect
        if a_key in out and b_key in out:
            rank = out[a_key].shape[0]
            scale = float(alpha) / rank if rank else 1.0
            if scale != 1.0:
                b = out[b_key]
                out[b_key] = (b.float() * scale).to(b.dtype)
            folded += 1
            scales.add(round(scale, 4))
    if folded:
        _log(f"LoRA: {folded} alpha(s) folded into the weights (alpha/rank = "
             f"{', '.join(str(s) for s in sorted(scales))})")
    return out, n


# ----------------------------------------------------------------------------
# LyCORIS LoKr. The update there is a Kronecker product: dW = w1 (x) w2. Neither peft nor
# diffusers knows how to apply that to a pipeline (not one occurrence of 'lokr' in
# loaders/lora_conversion_utils.py) -- but nothing stops us from MATERIALISING it and adding
# it to the weights. That is what a merge does, except it happens here, locally, without
# depending on a checkpoint merged by someone else.
#
# And the qkv rules out the other approach. On FLUX.2 the q/k/v is FUSED on the checkpoint
# side ([3d, d]) and SPLIT on the diffusers side (three [d, d]). A Kronecker product cannot
# be cut in three: with w1 [4,4] and w2 [3072,1024], w1's blocks are 3072 rows, not 4096.
# The materialised delta, on the other hand, cuts like any other matrix.
#
# An accepted trade-off, and an announced one: a merge is not an adapter. Changing the LoKr
# or its weight demands a transformer reload, where a PEFT LoRA is replaced hot.
# _apply_loras detects it and says so.
# ----------------------------------------------------------------------------

# Module prefixes the trainers use (ai-toolkit writes 'diffusion_model.').
# Deliberately NO 'lora_unet_': that kohya dialect replaces the dots with underscores in the
# module path, diffusers' converter would not recognise it, and pretending to support it
# would give a silently empty merge.
_LOKR_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")


def _strip_lokr_prefix(name):
    for p in _LOKR_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def _lokr_factor(mod, which):
    """(matrix, rank) of a LoKr factor. The FULL matrix when it is there (rank None:
    there is none), otherwise the product of its two reduced-rank factors."""
    full = mod.get(f"lokr_{which}")
    if full is not None:
        return full.to(torch.float32), None
    a, b = mod.get(f"lokr_{which}_a"), mod.get(f"lokr_{which}_b")
    if a is None or b is None:
        return None, None
    return a.to(torch.float32) @ b.to(torch.float32), int(a.shape[1])


def _lokr_scale(mod, rank):
    """The LyCORIS scale factor.

    When w1 AND w2 are full there is no rank: LyCORIS applies no scalar. The ai-toolkit
    files then write alpha = lora_dim (measured on SNOFS: 1e10), so alpha/rank would be 1.0
    too -- the two conventions agree, which is exactly why this can be decided without
    guessing. Otherwise alpha / rank, like peft (peft/tuners/lokr/layer.py:
    scaling = alpha / r).
"""
    if rank is None:
        return 1.0
    alpha = mod.get("alpha")
    return 1.0 if alpha is None else float(alpha) / float(rank)


def _lokr_delta(mod):
    """A LoKr module's float32 dW."""
    if "lokr_t2" in mod:
        raise ValueError("lokr_t2 (convolution factor) is not supported here")
    w1, r1 = _lokr_factor(mod, "w1")
    w2, r2 = _lokr_factor(mod, "w2")
    if w1 is None or w2 is None:
        raise ValueError("incomplete LoKr factors")
    return torch.kron(w1, w2) * _lokr_scale(mod, r1 if r1 is not None else r2)


def _merge_lokr(transformer, path, weight):
    """Merges a LoKr into the transformer's weights: W += weight * dW.

    MODULE BY MODULE. The complete delta of a SNOFS-9B weighs what the layers it touches
    weigh (~17 GB in bf16, twice that in float32): materialising it in one go would overflow
    the RAM for nothing, whereas one layer at a time caps at ~200 MB.

    The keys go through DIFFUSERS' Flux2 converter, the very one from_single_file uses: the
    renaming and the splitting of the fused qkv are therefore exactly those of the model's
    load and cannot diverge from it.

    Returns (n_merged, [what could not be]). Nothing is ever skipped silently: everything
    that does not find its target comes back in the second list.
"""
    from safetensors.torch import load_file
    from diffusers.loaders.single_file_utils import (
        convert_flux2_transformer_checkpoint_to_diffusers)
    sd = load_file(path)
    mods = {}
    for k, v in sd.items():
        base, _, suf = k.rpartition(".")
        if suf == "alpha" or suf.startswith(_LYCORIS_SUFFIXES):
            mods.setdefault(_strip_lokr_prefix(base), {})[suf] = v
    params = dict(transformer.named_parameters())
    hit, problems = 0, []
    for name in sorted(mods):
        try:
            delta = _lokr_delta(mods[name])
        except Exception as e:
            problems.append(f"{name}: {e}")
            continue
        # A dict with ONE entry: the converter works key by key (renamings +
        # handlers), so it returns 1 tensor here, or 3 when it is a fused qkv.
        for k, d in convert_flux2_transformer_checkpoint_to_diffusers(
                {name + ".weight": delta}).items():
            p = params.get(k)
            if p is None:
                problems.append(f"{k}: no such weight in the transformer")
            elif tuple(p.shape) != tuple(d.shape):
                problems.append(f"{k}: delta {tuple(d.shape)} vs weight {tuple(p.shape)}")
            else:
                with torch.no_grad():
                    # float32 for the addition: adding a small delta to a bf16 weight
                    # INSIDE bf16 loses the delta's low-order bits.
                    p.copy_((p.float() + d.to(p.device).float() * float(weight)).to(p.dtype))
                hit += 1
        del delta
    return hit, problems


def _lokr_set(loras):
    """The LoKr subset of a list of (path, weight): merged, not applied."""
    return [pw for pw in loras if _lycoris_algo(pw[0]) == "LoKr"]


def _peft_set(loras):
    """Everything that is not a LoKr, so what goes to peft. A LoHa deliberately STAYS in
    it: _sync_adapters refuses it by name, whereas filtering it silently here would make it
    disappear without a word."""
    return [pw for pw in loras if _lycoris_algo(pw[0]) != "LoKr"]


def _apply_lokrs_to(transformer):
    """Merges the LoKrs of LORAS into this transformer and updates _APPLIED_LOKRS.
    To be called right after the load and BEFORE the offload: the weights are still on the
    CPU, whole, with no accelerate hook on them."""
    global _APPLIED_LOKRS
    _APPLIED_LOKRS = []
    for p, w in _lokr_set(LORAS):
        t0 = time.time()
        try:
            n, problems = _merge_lokr(transformer, p, w)
        except Exception as e:
            _log(f"LoKr NOT merged, {os.path.basename(p)}: {type(e).__name__}: {e}")
            continue
        if problems:
            _log(f"LoKr {os.path.basename(p)}: {len(problems)} tensor(s) NOT merged, "
                 f"first: {problems[0]}")
        if not n:
            _log(f"LoKr {os.path.basename(p)}: nothing merged - the render will look "
                 f"exactly as if it were not selected")
            continue
        _log(f"LoKr merged into the weights: {os.path.basename(p)} @ {w} "
             f"-> {n} tensor(s) in {time.time() - t0:.1f}s")
        _APPLIED_LOKRS.append((p, w))


def _sync_adapters(pipe, wanted, applied, force=False, tag="LoRA"):
    """Synchronises a pipe's PEFT adapters with the `wanted` set, WITHOUT reloading the
    model. `applied` = the set really applied on this pipe (a list of (path, weight)).

    The transformer stays in VRAM; only the PEFT adapters move:
      - same files, different weights -> set_adapters (immediate)
      - a different LoRA set          -> unload_lora_weights + reloading the LoRAs (~1s)
    The derived pipes (from_pipe) share that transformer -> they follow automatically.
    Returns (ok, applied): ok=False on a failure (the caller decides: a full reload for the
    base, a hard error for editing), applied = the new set applied ([] on a failure).
"""
    # low_cpu_mem_usage=False on EVERY load (see _load_lora): diffusers' default
    # creates the adapter's layers on 'meta' then copies the weights into them. A DoRA whose
    # 'dora_scale' keys diffusers filters out then leaves parameters without data, and the
    # first move raises "Cannot copy out of meta tensor". With real tensors, a missing key
    # keeps its initial value.
    wanted = list(wanted)
    if not force and applied == wanted:
        return True, applied
    had_hooks = _offload_hooks(pipe)
    old_paths = [p for p, _ in applied]
    new_paths = [p for p, _ in wanted]
    try:
        if not force and old_paths and old_paths == new_paths:
            # Only the weights change -> an instant re-weighting.
            pipe.set_adapters(_lora_names(wanted), [float(w) for _, w in wanted])
            _log(f"{tag} weights updated in place (no reload): "
                 + ", ".join(f"{os.path.basename(p)}@{w}" for p, w in wanted))
            return True, wanted
        if old_paths or force:
            _clear_loras(pipe)
        names, weights = [], []
        for i, (p, w) in enumerate(wanted):
            if os.path.isfile(p):
                # A format peft cannot apply (LyCORIS LoKr/LoHa): it is named and we
                # move on. Letting it slip would be worse than an error: the render would
                # come out without the LoRA, identical to a render without it.
                why = _lora_unsupported(p)
                if why:
                    _log(f"{tag} SKIPPED, {os.path.basename(p)}: {why}")
                    continue
                an = f"cz_lora_{i}"
                _log(f"applying {tag}: {os.path.basename(p)} (weight {w})")
                # Pass the folder + weight_name (not the full path): otherwise
                # diffusers in offline mode (HF_HUB_OFFLINE) refuses with "must specify a
                # weight_name". Works online too, and with a direct local file.
                # From the 2nd adapter on, peft warns "Already found a peft_config":
                # stacking several LoRAs is precisely the point, so that message is
                # silenced.
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Already found a `peft_config`.*")
                    if _lora_needs_normalizing(p):
                        sd, nrn = _load_lora_normalized(p)
                        if nrn:
                            _log(f"{tag}: {nrn} key(s) converted from lora.down/up to the "
                                 f"PEFT dialect (otherwise peft would apply this LoRA only "
                                 f"partially, silently)")
                        _load_lora(pipe, sd, adapter_name=an)
                    else:
                        # Pass the folder + weight_name (not the full path): otherwise
                        # diffusers in offline mode (HF_HUB_OFFLINE) refuses with "must
                        # specify a weight_name". Works online too, and with a direct local
                        # file.
                        _load_lora(pipe, os.path.dirname(p) or ".",
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
        # diffusers removed the offload hooks before loading and did not get to put
        # them back: without this, the pipe stays on the CPU and EVERY later render fails,
        # including the ones that have nothing to do with this LoRA.
        if had_hooks and not _offload_hooks(pipe):
            restore_offload(pipe, f"a failed {tag} load")
        return False, []


def _apply_loras(pipe, force=False):
    """BASE LoRAs (txt2img/img2img): synchronises the pipe with LORAS through
    _sync_adapters. Returns True when applied, False on a failure (the caller falls back to
    a full reload)."""
    global _APPLIED_LORAS
    # A LoKr is MERGED into the weights: it can be neither removed nor re-weighted
    # without starting again from the original transformer. As soon as the requested set
    # differs from the one already inside, we say so and hand over to a full reload.
    want_lokr = _lokr_set(LORAS)
    if not force and want_lokr != _APPLIED_LOKRS:
        _log("LoKr selection changed -> full reload. A LoKr is merged INTO the weights "
             "(it is not a PEFT adapter), so it cannot be swapped or re-weighted in "
             "place: " + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in want_lokr)
                          or "none") + " wanted, "
             + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in _APPLIED_LOKRS)
                or "none") + " in the weights")
        return False
    ok, _APPLIED_LORAS = _sync_adapters(pipe, _peft_set(LORAS), _APPLIED_LORAS,
                                        force=force)
    if not ok:
        _log("falling back to a full reload")
    return ok


def _apply_edit_loras(pipe):
    """EDIT LoRAs: synchronises the edit pipe with EDIT_LORAS (or [] when the 'Edit
    LoRAs' box is unticked). A failure is a hard error: the user asked for that preset, and
    an edit WITHOUT it would be a false result.

    KLEIN DIVERGENCE. Upstream, editing and the base are TWO distinct models, so two
    independent adapter sets. Here it is the SAME object (see FORK.md, H):
      - the two sets were fighting over the `cz_lora_i` namespace -> "Adapter name
        cz_lora_0 already in use" as soon as a base LoRA AND an edit preset were both
        applied (a hard crash, no image at all);
      - and since `set_adapters` replaces the active list, applying the edit set
        silently DISABLED the base LoRAs.
    So the UNION (base + edit) is synchronised in a single call, with one source of truth
    `_APPLIED_LORAS`. `_APPLIED_EDIT_LORAS` stays the edit subset (cz_ui's contract: it
    shows it in generate_omni's log line).
"""
    global _APPLIED_LORAS, _APPLIED_EDIT_LORAS
    edit = list(EDIT_LORAS) if EDIT_LORAS_ENABLED else []
    # The fast mode's Lightning LoRA: stacked AFTER the presets (independent of the
    # 'Edit LoRAs' box, which only concerns the task presets).
    if EDIT_SPEED and EDIT_SPEED.get("path"):
        edit.append((EDIT_SPEED["path"], 1.0))
    # Union of base + edit, with no duplicate path (the first weight wins, the same
    # rule as the protocol's for `loras`).
    # The LoKrs are already IN the weights (merged at load time), they have no business in
    # an adapter set. An edit LoKr that is not in there cannot be applied hot: we say so,
    # rather than editing without it in silence.
    for p, w in _lokr_set(edit):
        if (p, w) not in _APPLIED_LOKRS:
            _log(f"edit LoKr {os.path.basename(p)} is NOT in the weights and cannot be "
                 f"merged on the fly; select it in Models > LoRA (a reload applies it)")
    # An edit LoRA that cannot be applied is a HARD ERROR, not a skip: the user asked
    # for that preset, and editing without it would return a false result. We raise BEFORE
    # peft, to give the reason in one sentence rather than forty 'size mismatch' lines.
    for p, _w in edit:
        why = _lora_unsupported(p)
        if why:
            _APPLIED_EDIT_LORAS = []
            raise RuntimeError(f"edit LoRA {os.path.basename(p)}: {why} "
                               f"Nothing was generated.")
    seen, wanted = set(), []
    for pw in _peft_set(list(LORAS) + edit):
        if pw[0] not in seen:
            seen.add(pw[0])
            wanted.append(pw)
    ok, applied = _sync_adapters(pipe, wanted, _APPLIED_LORAS, tag="edit LoRA")
    if not ok:
        _APPLIED_EDIT_LORAS = []
        raise RuntimeError("edit LoRA could not be applied on the edit pipe "
                           "(see log); nothing was generated")
    _APPLIED_LORAS = applied
    _APPLIED_EDIT_LORAS = [pw for pw in applied if pw in edit]


def _swap_transformer(pipe):
    """Replaces ONLY the transformer of the already cached pipeline: the VAE, the text
    encoder, the tokenizer and the scheduler stay in VRAM (they are most of the load time).
    Valid only with an identical base repo + EFFECTIVE offload.

    Returns True when the swap succeeded, False -> the caller does a full reload.
"""
    global _APPLIED_LORAS, _DERIVED
    t0 = time.time()
    old_t = _LOADED_KEY[1] if _LOADED_KEY else None
    # Switching to/from a GGUF changes the EFFECTIVE offload (a GGUF forces 'model')
    # -> the accelerate hooks and the placement differ: no tinkering, we reload.
    if _effective_offload(old_t) != _effective_offload(ZIMAGE_TRANSFORMER):
        _log("transformer swap skipped (GGUF changes the effective offload) -> full reload")
        return False
    try:
        _log(f"switching Klein transformer -> {ZIMAGE_TRANSFORMER or BASE_REPO} "
             "(keeping VAE + text encoder in VRAM)")
        new_t = _load_transformer()
        # A new transformer = new weights: the LoKrs merged into the old one are no
        # longer in it. So they are merged again BEFORE the placement, while it is still on
        # the CPU.
        _apply_lokrs_to(new_t)
        old = getattr(pipe, "transformer", None)
        off = _effective_offload()
        # Offload: the accelerate hooks are set on the components. They have to be
        # removed before the swap, or the new transformer has none and the old one keeps
        # its own.
        if DEVICE == "cuda" and off in ("model", "sequential"):
            try:
                pipe.remove_all_hooks()
            except Exception as e:
                _dbg(f"remove_all_hooks: {e}")
        try:
            pipe.register_modules(transformer=new_t)   # API diffusers (met a jour le config)
        except Exception:
            pipe.transformer = new_t
        # Free the OLD transformer BEFORE putting the new one on the GPU: otherwise
        # old + new + VAE/encoder exceed the VRAM -> a spill into shared RAM that never
        # recovers (measured on a multi-checkpoint XYZ grid on the studio side: 1.7 s/step
        # -> 300-600 s/step, then a crash). The derived pipes (from_pipe) point at the old
        # one too -> purge them first, or `del old` frees nothing (from_pipe is free, it
        # will be rebuilt).
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
                new_t.to(DEVICE)       # never a GGUF here (offload forced to 'model')
        # The LoRA adapters were applied on the old transformer -> to be reapplied.
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


def _hf_access_hint(repo, err):
    """An actionable message when the Hub REFUSED access to the repo, otherwise None (the
    original error travels up as is).

    The 4B is public; the 9B is gated (a non-commercial licence to accept). Without this
    message, a gated repo comes out as an HTTPError 401/403 in the middle of a
    huggingface_hub trace, and nothing says that all it takes is ticking a box + a token.
"""
    s = f"{type(err).__name__}: {err}"
    if not any(k in s for k in ("Gated", "gated", "401", "403", "restricted",
                                "awaiting a review", "Access to model")):
        return None
    if cz_core.hf_token_is_set():
        tok = ("A token IS being sent, so the missing piece is almost certainly the "
               "licence itself -- or the token belongs to another account.")
    else:
        tok = ("No token is being sent: set one (config 'hf_token', or "
               "'huggingface-cli login') with the SAME account that accepts the licence.")
    return (f"Hugging Face refused access to {repo}. That repo is gated: open "
            f"https://huggingface.co/{repo} and accept its licence. {tok} "
            f"Original error -- {s}")


def _ensure_base():
    """Loads (when needed) the base txt2img pipeline. Handles the single-file/GGUF
    transformer and the offload. Cached by (repo, transformer, offload).

    Two hot swaps avoid a full reload (transformer + VAE + text encoder, tens of seconds):
      - different LoRAs            -> _apply_loras (the PEFT adapters alone)
      - a different transformer, same base repo + offload -> _swap_transformer.
"""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _BASE_SCHED_CONFIG, _APPLIED_LORAS
    global _TEXT_ENCODER_ACTIVE
    key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE)
    _dbg(f"_ensure_base key={key} cached={_LOADED_KEY}")
    if _BASE_PIPE is not None and _LOADED_KEY == key:
        if _apply_loras(_BASE_PIPE):
            _dbg("base pipeline: reusing cached (no reload)")
            return _BASE_PIPE
        _dbg("base pipeline: LoRA hot-swap failed -> free + reload")
        free_vram()
    elif _BASE_PIPE is not None:
        # Only the transformer changes (same base repo + same offload)? -> reload the
        # transformer ONLY and keep VAE + text encoder in VRAM.
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
    # A replacement encoder: checked on the config then loaded with the repo's class.
    # An encoder that does not suit (the repo went from 4B to 9B since the choice, an
    # unreadable folder) is dropped WITH a log line, and the metadata says so.
    _TEXT_ENCODER_ACTIVE = ""
    if TEXT_ENCODER:
        _why = _text_encoder_problem(TEXT_ENCODER)
        if not _why:
            try:
                kwargs["text_encoder"] = _load_monitor(
                    f"text encoder {_encoder_label(TEXT_ENCODER)}",
                    lambda: _load_text_encoder(TEXT_ENCODER))
                _TEXT_ENCODER_ACTIVE = TEXT_ENCODER
            except Exception as e:
                _why = f"it failed to load ({type(e).__name__}: {e})"
        if _why:
            _log(f"text encoder {_encoder_label(TEXT_ENCODER)} NOT used: {_why}. "
                 f"{BASE_REPO}'s own encoder runs instead; the image metadata says so "
                 f"(text_encoder_not_applied).")
        else:
            _log(f"text encoder: {_encoder_label(TEXT_ENCODER)} replaces {BASE_REPO}'s "
                 f"own (tokenizer, VAE and transformer unchanged)")
    _need = _base_vram_need_gb()
    _off_label = (f"auto->{_resolve_auto()}" if OFFLOAD_MODE == "auto" else OFFLOAD_MODE)
    _log(f"loading FLUX.2 Klein base: {BASE_REPO} (offload={_off_label}, dtype=bf16"
         + (f", ~{_need:.0f} GB of weights" if _need else "")
         + ") ... first run downloads it from HF, then cached")
    try:
        pipe = _load_monitor(f"FLUX.2 Klein base {BASE_REPO}",  # noqa: E128
                             lambda: Flux2KleinPipeline.from_pretrained(BASE_REPO, torch_dtype=DTYPE,
                                                                        **kwargs))
    except Exception as e:
        hint = _hf_access_hint(BASE_REPO, e)
        if hint:
            raise RuntimeError(hint) from e
        raise
    # Capture the scheduler's native (flow-matching) config -> the base for building
    # the other samplers (euler/dpm2a/dpmpp2m) without losing shift/flow params.
    try:
        _BASE_SCHED_CONFIG = dict(pipe.scheduler.config)
    except Exception:
        _BASE_SCHED_CONFIG = None
    # Qwen-Image LoRAs (on the base's transformer -> shared by the derived pipes).
    # force=True: a new pipe, no adapter applied -> we (re)apply everything.
    _APPLIED_LORAS = []
    # The same window as the LoKrs: still on the CPU, with no accelerate hook set.
    _trim_text_encoder(pipe)
    # The LoKrs BEFORE any move/offload: the transformer is still on the CPU, in one
    # piece, with no accelerate hook -- that is the only window where merging into the
    # weights is simple and safe.
    _apply_lokrs_to(pipe.transformer)
    if LORAS:
        _apply_loras(pipe, force=True)
    # Attention slicing: SET PER CALL through _set_slicing (according to the
    # resolution processed), NOT at load time. In tiles/at 1024 -> slicing OFF = native SDPA
    # attention, fast (like ComfyUI). Whole-image 2K+ -> slicing ON to avoid the 32 GB VRAM
    # spill.
    # enable_*_cpu_offload handles the device itself -> do NOT call .to(cuda) then.
    # IMPORTANT: a quantized GGUF transformer does NOT move onto the GPU through .to(cuda)
    # (offload=none) nor in sequential -> it stays on the CPU = ULTRA slow (empty VRAM,
    # ~500s/step). Only enable_model_cpu_offload (accelerate) places it properly on the GPU
    # during the forward. So 'model' is forced for a GGUF base, whatever the UI/config says.
    _off = _effective_offload()
    _base_off = _resolve_auto() if OFFLOAD_MODE == "auto" else OFFLOAD_MODE
    if _off != _base_off:
        if _is_gguf_path(ZIMAGE_TRANSFORMER):
            _log(f"GGUF base: offload '{_base_off}' forced to '{_off}' (a GGUF does not "
                 f"run on the GPU in none/sequential -> it would stay on CPU, ~500s/step)")
        else:
            _need, _have = _base_vram_need_gb(), _total_vram_gb()
            _log(f"offload '{_base_off}' forced to '{_off}': {BASE_REPO} needs about "
                 f"{_need:.0f} GB on the GPU and this card has {_have:.1f} GB. Loading it "
                 f"whole would die at the first diffusion step on a CUDA error that does "
                 f"not even name the VRAM. '{_off}' streams the weights instead -- slower "
                 f"per image, but it runs. Set `default_cpu_offload` to silence this.")
    if DEVICE == "cuda" and _off == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and _off == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    # VAE tiling/slicing: essential for img2img/upscale. Qwen-Image is big (~20B
    # transformer + text encoder) -> without tiling the VAE can overflow the VRAM (a spill
    # into shared RAM = very slow). Tiling the VAE caps that peak (like ComfyUI's "tiled
    # decode"). The VAE is shared by the derived pipes.
    try:
        pipe.vae.config.force_upcast = False   # the VAE in bf16 (fp32 is slow on Blackwell) -- ALWAYS
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception as e:
        _dbg(f"VAE tiling not available: {e}")
    _apply_sampler(pipe)   # applies the chosen sampler (euler by default) to the base pipe
    _BASE_PIPE = pipe
    _DERIVED = {"txt2img": pipe}
    _LOADED_KEY = key
    _log(f"FLUX.2 Klein base ready in {time.time() - t0:.1f}s (sampler={SAMPLER}/{SCHEDULE})")
    return pipe


def get_pipe(kind="img2img"):
    """Returns the requested pipeline. txt2img/img2img/inpaint derive from the base through
    from_pipe (shared weights). Omni needs extra components (SigLIP) -> loaded separately
    from a dedicated Omni model (CONFIG['zimage_omni_model'])."""
    if kind == "omni":
        # klein: multi-reference editing IS the base pipeline (`image` takes a list of
        # PIL images). No second model to load -> no duplicate VRAM.
        _dbg("get_pipe('omni'): pipeline de base (multi-reference natif)")
        return _ensure_base()
    base = _ensure_base()
    # A sampler/schedule change asked for DURING a render was only recorded
    # (see _reapply_sampler_all): this is where it lands, between two
    # generations, with _GPU_LOCK held by the caller.
    _apply_sampler_if_dirty()
    if kind in _DERIVED:
        _dbg(f"get_pipe('{kind}'): reuse derived")
        return _DERIVED[kind]
    # `Flux2KleinPipeline` does NOT expose `strength`: its `image` is a reference
    # conditioning (Kontext-style), not a noised start. So img2img goes through the INPAINT
    # pipeline with a full white mask (injected by _qwen_call). A single derived object
    # serves both -> it is shared under both cache keys.
    from diffusers import Flux2KleinInpaintPipeline
    cls = {"img2img": Flux2KleinInpaintPipeline, "inpaint": Flux2KleinInpaintPipeline}.get(kind)
    if cls is None:
        return base
    twin = "inpaint" if kind == "img2img" else "img2img"
    if twin in _DERIVED:
        _dbg(f"get_pipe('{kind}'): reuse '{twin}' (same Flux2KleinInpaint pipeline)")
        _DERIVED[kind] = _DERIVED[twin]
        return _DERIVED[kind]
    _log(f"deriving {kind} pipeline (shared weights, no extra VRAM)")
    # A GGUF transformer is QUANTIZED: it cannot be recast to a dtype (.to(DTYPE)
    # raises "Casting a quantized model is unsupported"). So the bf16 recast is skipped in
    # that case (the compute_dtype is bf16 already). Otherwise (full bf16): a defensive
    # Blackwell recast (some from_pipe calls upcast to float32 -> very slow without fp32
    # tensor cores).
    quantized = bool(ZIMAGE_TRANSFORMER) and ZIMAGE_TRANSFORMER.lower().endswith(".gguf")
    try:
        # A quantized GGUF: torch_dtype=None EXPLICITLY -> otherwise from_pipe puts
        # float32 by default and casts the quantized model -> ValueError "Casting a
        # quantized model".
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
    _apply_sampler(p)   # the same sampler as the base (in case from_pipe recreates the scheduler)
    # Speed diagnosis: if the derived pipe is NOT on cuda -> img2img/refine runs on the
    # CPU = ultra slow. So it is forced onto DEVICE in full-VRAM mode (offload handles
    # itself).
    # NB: the EFFECTIVE offload (a GGUF base forces 'model' even when the UI says 'none'):
    # under offload, a transformer "on the CPU" is normal -> a .to(cuda) would break the
    # hooks.
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
    """FLUX.2 Klein multi-reference editing: edits one (or several, up to 4) input
    image(s) according to the instruction prompt. Keeps upstream's signature (cz_ui).

    On klein it is the BASE pipeline that edits (`image` takes a list of PIL images): no
    second model, no duplicate VRAM, no extra loading time.
    `negative` is accepted for compat but has NO EFFECT (a distilled model, see _cfg) -
    cz_protocol announces it through supports.negative = False.
    width/height are ignored by default (editing preserves the input's dimensions);
    honor_size=True passes them to the pipe. The edit LoRAs (EDIT_LORAS, the 'Edit LoRAs'
    box) are applied hot here, on the same transformer as the base.
"""
    refs = [r.convert("RGB") for r in (refs or []) if r is not None]
    if not refs:
        raise ValueError("Edit needs at least one input image.")
    pipe = get_pipe("omni")
    _apply_edit_loras(pipe)
    # Fast mode: its steps/guidance win over Settings; a caller that set them
    # explicitly (protocol spec.steps / spec.guidance) keeps the upper hand.
    if EDIT_SPEED:
        if not steps_explicit:
            steps = EDIT_SPEED["steps"]
        if guidance is None:
            guidance = EDIT_SPEED["guidance"]
    g = float(GUIDANCE) if guidance is None else float(guidance)
    lora_info = ", edit LoRA " + "+".join(os.path.basename(p) for p, _ in _APPLIED_EDIT_LORAS) \
        if _APPLIED_EDIT_LORAS else ""            # the presets + the Lightning LoRA really applied
    _log(f"edit: {len(refs)} image(s), {int(steps)} steps, cfg {g:.1f}{lora_info} ...")
    _progress(0.1, f"Editing ({len(refs)} image(s))...")
    _set_slicing(pipe, max(max(r.size) for r in refs))
    t0 = time.time()
    # Flux2KleinPipeline takes `list[PIL] | PIL`: the list is passed as is as soon as
    # there is more than one reference.
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
    """Qwen-Image txt2img: generates an image from a prompt. Real CFG through
    true_cfg_scale (= the guidance slider, ~4.0), ~30-50 steps advised. The negative prompt
    works thanks to the real CFG (see _cfg)."""
    pipe = get_pipe("txt2img")
    w = round_to_multiple(int(width))
    h = round_to_multiple(int(height))
    _log(f"txt2img: {w}x{h}, {int(steps)} steps, cfg {GUIDANCE:.1f} ...")
    _dbg(f"txt2img seed={seed} dtype=bf16 device={DEVICE} offload={OFFLOAD_MODE} "
         f"transformer={'single-file' if ZIMAGE_TRANSFORMER else 'repo'}")
    if DEVICE == "cuda":
        _dbg(f"VRAM before: alloc={torch.cuda.memory_allocated()/1024**3:.2f} Go")
    _progress(0.1, f"Generating {w}x{h} ({int(steps)} steps)...")
    t0 = time.time()
    # Two attempts at most: when the VRAM guard fires at the first step ('none' mode
    # too optimistic), _consume_vram_downgrade switches to 'model' and we replay.
    for _attempt in (0, 1):
        _set_slicing(pipe, max(w, h))
        img = _qwen_call(
            pipe,
            prompt=prompt or "",
            width=w, height=h,
            num_inference_steps=int(steps),
            generator=_make_generator(seed),
            **_cfg(negative_prompt),
            **_vram_guard_kwargs(),
        ).images[0]
        if not _consume_vram_downgrade():
            break
        pipe = get_pipe("txt2img")   # reload with the downgraded offload
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
    """Sets the forced ratio for upscale/img2img: 'W:H' / 'WxH' (e.g. '13:19',
    '832x1216') or '' to turn it off (the native ratio is preserved). Driven by the UI
    radio."""
    global FORCE_RATIO
    FORCE_RATIO = (spec or "").strip()
    _log(f"force ratio -> {FORCE_RATIO or '(off, ratio natif preserve)'}")


def set_force_ratio_mode(mode):
    """'crop' (a center crop) or 'extend' (outpainting the missing bands)."""
    global FORCE_RATIO_MODE
    FORCE_RATIO_MODE = "extend" if str(mode or "").strip().lower() == "extend" else "crop"
    _log(f"force ratio mode -> {FORCE_RATIO_MODE}")


def _parse_ratio(spec):
    """(w, h) from 'W:H', 'WxH', or a '832 x 1216 | 13:19' label; otherwise None."""
    import re
    if not spec:
        return None
    m = re.search(r"(\d+)\s*[:xX×]\s*(\d+)", str(spec))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if a > 0 and b > 0 else None


def _crop_to_ratio(image, ratio_w, ratio_h):
    """Centre-crops the image to the ratio_w:ratio_h ratio, keeping the largest area."""
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur > target:                       # too wide -> cut the sides
        nw = max(1, int(round(h * target)))
        x0 = (w - nw) // 2
        return image.crop((x0, 0, x0 + nw, h))
    nh = max(1, int(round(w / target)))    # trop haut -> couper haut/bas
    y0 = (h - nh) // 2
    return image.crop((0, y0, w, y0 + nh))


def _extend_to_ratio(image, ratio_w, ratio_h, prompt, steps, seed):
    """Brings the image to the target ratio by EXTENDING it (outpaint) instead of
    cropping: symmetric bands are added on the missing axis and filled by the model through
    outpaint_directions -- the centre keeps its full resolution (only the bands are
    generated, with the diffusion bounded to ~1 MP then recomposed).

    Anti 'banding': a light img2img pass (EXTEND_DENOISE) runs on the extended image, but
    ONLY the bands + a feathered transition margin are pasted back from that pass -- the
    original centre stays PIXEL FOR PIXEL intact (the pass harmonises exposure/texture at
    the seams without ever retouching the image).
"""
    from PIL import ImageDraw, ImageFilter
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur < target:                       # too narrow -> widen left + right
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
        # Paste-back mask: white = take the harmonised pass (the bands + a transition
        # margin STRADDLING the seam), black = keep the original. The margin reaches into
        # the original image then is feathered -> a blended join, an intact centre.
        ox, oy = (out.width - w) // 2, (out.height - h) // 2
        m = max(24, int(0.05 * min(out.size)))       # a transition ~5% of the short side
        mx, my = (m if ox > 0 else 0), (m if oy > 0 else 0)   # a margin on the seam side ONLY
        mask = Image.new("L", out.size, 255)
        ImageDraw.Draw(mask).rectangle(
            [ox + mx, oy + my, ox + w - mx, oy + h - my], fill=0)
        mask = mask.filter(ImageFilter.GaussianBlur(max(8, m // 3)))
        out = Image.composite(refined, out, mask)
    return out


def _reframe_canvas(image, ratio_w, ratio_h, overlap=8):
    """Places the image in a larger canvas at the target ratio (expansion on 1 axis),
    + a mask (white = to fill, black = to keep, with a small overlap)."""
    from PIL import ImageDraw
    image = image.convert("RGB")
    w, h = image.size
    r = ratio_w / ratio_h
    # Aligned on 32 (patch 2 x VAE 16): avoids conv errors (no engine).
    if w / h < r:  # trop etroit -> elargir
        nw, nh = round_to_multiple(int(round(h * r)), 32), round_to_multiple(h, 32)
    else:          # too wide -> grow in height
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
    """Inpaint: regenerates the white area of the mask according to the prompt
    (ZImageInpaintPipeline). background + mask = PIL (L: white = to change)."""
    orig = background.convert("RGB")
    full_mask = mask
    # Diffusion bounded to ~1 MP (the model's sweet spot), then recomposed at full
    # resolution.
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
    # Recompose: outside the mask keeps the full resolution; the join is feathered.
    out = _composite_back(out, orig, full_mask, orig_size,
                          feather=max(2, int(min(orig_size) * 0.01)))
    _log(f"inpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# The Z-Image model's "sweet spot" target resolution (~1 MP, like the txt2img ratios).
# The reframe aims at that budget so as NOT to blow up the pixel count (a 2-3 MP output that
# leaves the training zone -> slow and degraded quality).
MODEL_TARGET_PX = 1024 * 1024


def _ratio_canvas(ratio_w, ratio_h, target_px=MODEL_TARGET_PX):
    """A canvas's size (multiples of 32) at the given ratio, around target_px pixels."""
    r = float(ratio_w) / float(ratio_h)
    nh = (target_px / r) ** 0.5
    nw = nh * r
    return round_to_multiple(int(round(nw)), 32), round_to_multiple(int(round(nh)), 32)


def _cap_work_res(image, mask, max_px=MODEL_TARGET_PX):
    """Bounds the working resolution for the diffusion: when image > max_px, returns a
    reduced version (multiples of 32) of (image, mask) + the original size to recompose
    afterwards. Avoids running the model far above its sweet spot (~1 MP) -> faster and
    better quality."""
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
    """Recomposes at the original resolution: the masked area (white) comes from `result`
    (scaled back up to orig_size), the rest from `original` -> everything outside the mask
    keeps the starting image's full resolution. `feather` (px) blurs the mask to blend the
    join (a gradual original <-> generated transition, no hard line)."""
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
    """Crops the image to the target ratio while bounding the output to the model's sweet
    spot (~1 MP) -> no more pixel-count explosion.
      fit='contain' : the whole image fits inside the canvas (without enlarging it), and the
                      added edges are filled by Z-Image (outpaint).
      fit='cover'   : the image fills the canvas at the ratio then is center-cropped (no
                      outpaint, a plain reframe/crop).
"""
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
    # contain -> the original is fitted without being enlarged, then the edges are
    # outpainted.
    from PIL import ImageFilter
    scale = min(nw / w, nh / h, 1.0)
    rw2, rh2 = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((rw2, rh2), Image.LANCZOS) if (rw2, rh2) != (w, h) else img
    ox, oy = (nw - rw2) // 2, (nh - rh2) // 2
    # Edges = a blurred extension of the edge colours (blurred edge fill, like the
    # outpaint) rather than a grey -> exposure continuity; it shows through when
    # strength < 1.0.
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
    """Compat (CLI --reframe and existing calls): a reframe in 'contain' mode (outpaint),
    bounded to the model's sweet spot."""
    return reframe(image, ratio_w, ratio_h, "contain", prompt, steps, seed)


def outpaint_directions(image, mask, directions, prompt, steps, seed, strength=1.0, expand=0.3):
    """Directional outpaint (Fooocus-style): enlarges the image in the chosen directions
    among left/right/top/bottom, each by `expand` (a fraction of the original dimension), by
    replicating the edge pixels (mode 'edge'), then has the added bands filled by Z-Image
    (ZImageInpaintPipeline). A painted `mask` (L, white = to change) is optional: it is kept
    over the original area and combined with the added bands (white)."""
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
    # Dilate the area to generate a little towards the inside -> the model regenerates
    # a thin transition band that joins up with the original (avoids a hard seam).
    from PIL import ImageFilter
    k = max(3, (int(min(full_size) * 0.02) // 2) * 2 + 1)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(min(k, 15)))
    # "Blurred edge fill": the area to generate is filled with a BLURRED version of
    # the edge extension (the same colours/tone as the original) instead of a sharp
    # replicated edge. With strength < 1.0 that blur shows through -> exposure continuity
    # (no lighter band any more) and the model adds the detail on top.
    blur_r = max(8, int(min(full_size) * 0.03))
    canvas = Image.composite(canvas.filter(ImageFilter.GaussianBlur(blur_r)), canvas, mask_img)
    # Diffusion bounded to ~1 MP (the sweet spot), then recomposed: the centre (the
    # original image) keeps its full resolution, only the added edges are generated.
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
    """A Qwen-Image img2img pass over the whole image (or over one tile). The slicing is
    set according to the size really processed: a 1024 tile -> OFF (fast), whole 2K+ -> ON.
    IMPORTANT: width/height = the image's size (aligned on 16) are passed. Otherwise
    Qwen-Image img2img falls back to its default (height = default_sample_size *
    vae_scale_factor = 1024) and RESIZES the input to 1024x1024 -> the ratio is crushed (a
    bug). Forcing the input's dimensions preserves the original ratio in upscale/img2img."""
    w = round_to_multiple(image.width, 16)
    h = round_to_multiple(image.height, 16)
    # Two attempts at most: the VRAM guard at the first step (see generate), then a retry in 'model'.
    for _attempt in (0, 1):
        _set_slicing(pipe, max(image.size))   # to set again on the pipe the retry reloaded
        out = _qwen_call(
            pipe,
            prompt=prompt or "",
            image=image,
            width=w, height=h,
            strength=float(denoise),
            num_inference_steps=int(steps),
            generator=_make_generator(seed),
            **_cfg(None),
            **_vram_guard_kwargs(),
        ).images[0]
        if not _consume_vram_downgrade():
            return out
        pipe = get_pipe("img2img")   # reload with the downgraded offload
    return out


def _feather_mask_np(th, tw, overlap, left, right, top, bottom):
    """A (th, tw, 1) mask with a linear ramp on the edges that adjoin another tile."""
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
    """A Z-Image pass in tiles with feathered recomposition (Ultimate SD Upscale style).
    Caps the VRAM peak (one tile at a time) and makes 4K+ possible without seams.
    The same linear ramp + overlap-add as esrgan_upscale, but at scale 1 on PIL."""
    w, h = image.size
    tile = round_to_multiple(tile)                       # a multiple of 16 for the VAE
    overlap = max(0, min(int(overlap), tile - 16))
    if w <= tile and h <= tile:
        # A single tile = the whole image -> no duplication possible: the requested
        # denoise.
        return _refine_whole(pipe, image, denoise, steps, prompt, seed)
    # Anti-duplication 1: an empty prompt per tile (the global prompt describes the
    # whole composition).
    prompt = _tile_prompt(prompt)
    if not (prompt or "").strip():
        _log("refine tiled: empty prompt per tile (anti-duplication; rule refine_tile_prompt).")
    # Anti-duplication 2 (a safety net): at a high denoise each tile can still drift.
    denoise = float(denoise)
    if _TILE_DENOISE_CAP > 0 and denoise > _TILE_DENOISE_CAP:
        _log(f"refine tiled: denoise {denoise:.2f} > the cap {_TILE_DENOISE_CAP:.2f} -> "
             f"lowered to {_TILE_DENOISE_CAP:.2f} (refine_tile_denoise_cap rule).")
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
# Orchestration: process_one, the txt2img batch (run/_gen_meta stay in app.py because
# run emits gr.Error for the UI).
# ----------------------------------------------------------------------------
@_gpu_serial
def process_one(image, esrgan_model, factor, denoise, steps, prompt, seed, tile, overlap,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                do_esrgan=True, refine_first=False, apply_force_ratio=False):
    """Pipeline over one PIL Image, returns (image, timings_dict).
    do_esrgan=False -> pure img2img (skips the ESRGAN stage, refines the native image).
    refine_first=True -> refine THEN ESRGAN (the diffusion runs at the native resolution =
    far faster), instead of ESRGAN THEN refine (detail at high resolution).
    apply_force_ratio=True + FORCE_RATIO set -> brings the INPUT to the chosen ratio before
    processing: FORCE_RATIO_MODE 'crop' = a center crop (Fooocus-style), 'extend' =
    outpaints the missing bands (nothing is lost). Otherwise: the native ratio is preserved.
"""
    timings = {"esrgan": 0.0, "refine": 0.0}
    image = image.convert("RGB")
    if apply_force_ratio and FORCE_RATIO:
        r = _parse_ratio(FORCE_RATIO)
        if r:
            _before = image.size
            if FORCE_RATIO_MODE == "extend":
                # max(6, steps): outpainting the bands stays correct even when the
                # upscale runs as pure ESRGAN (steps/denoise at ~0).
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
        # The target = the factor applied to the original size (order-independent).
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
        # Anti-crash guard rail: a whole-image refine that is too large (4K+) -> auto-tiling.
        if rt <= 0 and max(rw, rh) > _AUTO_TILE_ABOVE:
            rt = _pick_refine_tile(rw, rh, int(refine_overlap) or 64)
            _log(f"refine: image {rw}x{rh} > {_AUTO_TILE_ABOVE}px -> auto-tiling (tile {rt}) "
                 "to avoid the VRAM spike (rules: auto_refine_tile_above, auto_refine_tile)")
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
        # refine on the native image (fast) then the ESRGAN enlargement.
        if do_refine:
            result = _refine_stage(result)
        if use_esrgan:
            result = _esrgan_stage(result)
    else:
        # the classic order: ESRGAN (the detailer) then refine at the enlarged resolution.
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
    """Generates an image (Z-Image txt2img) then, when upscale=True, runs it through the
    ESRGAN + refine pipeline. Returns (image, timings_dict)."""
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


# The modes where the EDIT LoRA set is really applied (the omni branch of _ui_generate
# and the protocol's 'edit' op). Elsewhere it is not, and saying so would be lying.
_EDIT_MODES = ("omni", "edit")

# ----------------------------------------------------------------------------
# INPUT image(s) in the metadata. An img2img, an inpaint or an edit is defined as much by
# its input as by its prompt: without it, the file does not reproduce from itself.
#
# The NAME by default, not the path. The PNG travels (Civitai, forums, a client) whereas the
# sidecar stays local: a full path would export the disk's tree and the Windows session name
# with it. And on the UI side it would be worth nothing anyway -- Gradio drops uploads in a
# temporary folder where only the BASE NAME carries the file's original name. 'full' only
# makes sense for inputs taken from a folder (batch processing), where the path still exists
# tomorrow.
# ----------------------------------------------------------------------------
METADATA_SOURCE = str(CONFIG.get("metadata_source", "name") or "name").strip().lower()


def _source_path_of(x, _depth=0):
    """File path of an image input, or None when it cannot be known.

    Accepts a path, a PIL opened from a file (.filename), or a gr.ImageEditor value
    ({background, composite, layers}). The background is tried BEFORE the composite:
    after a crop the composite is a brand new image, with no name.
"""
    if not x or _depth > 2:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("path", "name", "background", "composite", "image"):
            p = _source_path_of(x.get(k), _depth + 1)
            if p:
                return p
        return None
    p = getattr(x, "filename", None)
    return p if isinstance(p, str) and p else None


def source_meta(items, key="source"):
    """A metadata fragment naming the input image(s), or {} when it is not known.
    Nothing rather than an invented name: a wrong piece of metadata is worse than an absent
    one."""
    if METADATA_SOURCE in ("off", "none", "no", "0", "false"):
        return {}
    full = METADATA_SOURCE in ("full", "path", "abs")
    vals = []
    for it in (items if isinstance(items, (list, tuple)) else [items]):
        p = _source_path_of(it)
        if p:
            vals.append(os.path.abspath(p) if full else os.path.basename(p))
    if not vals:
        return {}
    return {key: vals[0] if len(vals) == 1 else vals}


def _gen_meta(mode, prompt, negative="", seed=None, steps=None, guidance=None,
              size=None, model=None, styles=None, extra=None):
    """Builds the generation metadata dict (for the sidecar/PNG)."""
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
    # Names of the applied styles (on top of the keywords already injected into the
    # prompt).
    _styles = [s for s in (styles or []) if s and s not in ("None", "none")]
    if _styles:
        m["styles"] = _styles
    m["sampler"] = f"{SAMPLER}/{SCHEDULE}"
    m["model"] = model or (ZIMAGE_TRANSFORMER or BASE_REPO)
    # A single-file only replaces the TRANSFORMER: the VAE, the text encoder and the
    # architecture config come from the base repo, and 4B/9B are not interchangeable.
    # Without it, the image is not reproducible.
    if ZIMAGE_TRANSFORMER:
        m["base_repo"] = BASE_REPO
    # A replacement encoder: the one that REALLY ran, by its folder name. Requested
    # but dropped at load time = the image comes from the base repo's encoder, and the one
    # that did not serve is named separately.
    if _TEXT_ENCODER_ACTIVE:
        m["text_encoder"] = _encoder_label(_TEXT_ENCODER_ACTIVE)
    elif TEXT_ENCODER:
        m["text_encoder_not_applied"] = _encoder_label(TEXT_ENCODER)
    # What was REALLY applied, not what was asked for. A LoKr is merged into the
    # weights (_APPLIED_LOKRS) and does not appear among the PEFT adapters; and now that
    # LoRAs can be dropped along the way (wrong variant, unsupported LyCORIS, a quantized
    # build, a missing file), listing LORAS would amount to signing an image with a LoRA it
    # does not carry.
    applied = list(_APPLIED_LORAS) + list(_APPLIED_LOKRS)
    if applied:
        m["loras"] = [f"{os.path.basename(p)}@{w}" for p, w in applied]
    missing = [pw for pw in LORAS if pw not in applied]
    if missing:
        m["loras_not_applied"] = [f"{os.path.basename(p)}@{w}" for p, w in missing]
    # The EDIT set: distinct from the base set, and it is the one that shapes an
    # edit's result. It was missing from the metadata, so an edit did not reproduce from its
    # own file.
    # ... and ONLY on an edit: _APPLIED_EDIT_LORAS outlives the edit that applied it, so a
    # later txt2img would claim a set it never carried. A wrong piece of metadata is worse
    # than an absent one.
    if mode in _EDIT_MODES:
        if _APPLIED_EDIT_LORAS:
            m["edit_loras"] = [f"{os.path.basename(p)}@{w}"
                               for p, w in _APPLIED_EDIT_LORAS]
        if EDIT_SPEED and EDIT_SPEED.get("name"):
            m["edit_speed"] = EDIT_SPEED["name"]
    if extra:
        m.update(extra)
    return m
