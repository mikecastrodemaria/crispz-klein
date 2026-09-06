# Changelog

All notable changes to crispz-klein. One versioned entry per feature.
The app version lives in `cz_core.py` (`APP_VERSION`) and is shown in the browser tab title.

Entries at 1.17.0 and below are inherited from crispz-qwen-edit / crispz-studio and
describe the Qwen-Image engine. The fork to FLUX.2 Klein is documented in
[FORK.md](FORK.md).

## 1.18.5 — Hide what klein cannot do, stop advertising what it now can

Two stale labels, both mine, both from the moment the catalogue was empty:

- **"Edit LoRA presets (none published for FLUX.2 yet)"** — no longer true since
  1.18.2, and the help still listed the Qwen presets (Photo-to-Anime, Any-Light,
  Upscaler, Multiple-Angles) that cannot load here. It now says what it actually
  guarantees: only LoRAs verified to load on FLUX.2 Klein are listed.
- **"Edit speed"** was still offered with help describing Lightning 2509/2511 and
  Rapid-AIO. On klein there is nothing to accelerate — the model is already
  distilled to 4 steps with no CFG, and no FLUX.2 Lightning LoRA exists — so the
  dropdown only ever held "Off". The whole row is now HIDDEN when the only choice
  is "Off", the way crispz-krea2 hides what its family cannot do. The components
  are built and merely invisible, so the handlers stay wired: nothing to re-cable
  the day an acceleration LoRA ships.

`caps.edit_fast` still reports `["off"]` rather than disappearing, so family
clients keep reading the same field.

## 1.18.4 — Say it once, and stop calling things 'zimage'

The 4B/9B guard added in 1.18.3 was right but unusable in practice: it repeated a
four-line paragraph for EVERY skipped file. On a library with eleven klein-9B
checkpoints that is eleven paragraphs, and the useful lines drown.

Now: one short line per file (debug level), then ONE summary that carries the
instructions — *"11 checkpoint(s) skipped: they are FLUX.2-klein-9B builds and
this install runs FLUX.2-klein-4B. To use them, point 'klein_model' at the
matching base repo. Note: FLUX.2-klein-9B is NON-COMMERCIAL..."*. Three lines
instead of eleven paragraphs. The variant name is also written properly
(`FLUX.2-klein-9B`, not `FLUX.2-klein 9B`).

That message also exposed a naming problem: it told the user to set
**`zimage_model`** — a Z-Image leftover from crispz-studio, meaningless in a
FLUX.2 fork. The config and env keys are renamed:

  klein_model        (was zimage_model,       env KLEIN_MODEL)
  klein_transformer  (was zimage_transformer, env KLEIN_TRANSFORMER)
  --klein-model / --klein-transformer  (CLI)

Old names are still READ, and reading one logs a rename hint; the old CLI flags
stay as hidden aliases so no existing script breaks. The Python module variables
keep their names (`ZIMAGE_TRANSFORMER`, `set_zimage_model`…) — cz_ui, cz_cli and
cz_protocol import them, that is the module's API contract.

## 1.18.3 — Tell 4B from 9B before loading, not after

A klein-9B checkpoint loaded into the 4B pipeline died deep inside diffusers,
after reading gigabytes, on an unreadable message: *"expected shape
[18432, 3072], but got [24576, 4096]"*. The architecture guard could not catch it
— **4B and 9B share the architecture and the tensor names**; only the hidden
dimension differs (3072 vs 4096).

It is now read from the **header**, before any weight: the signature is
`double_stream_modulation_img` (`.lin.` in the original/ComfyUI layout,
`.linear.` in the diffusers one), whose second axis IS the hidden dim. The
expected value comes from the configured base repo
(`attention_head_dim * num_attention_heads`), so the guard is **base-relative and
symmetric**: point `klein_model` at the 9B repo and it is the 4B files that get
refused. A base whose dimension cannot be determined filters nothing — the house
rule is to never discard a model on a doubt.

The refusal names both variants, the two dimensions, and what to do about it. On
a 9B file it also flags that **klein-9B is under a non-commercial licence**,
unlike the 4B.

Same check on the GGUF path (`_gguf_hidden_dim`), where a 9B quant hit exactly
the same wall.

Measured on a real 21-file library: 7 loadable, 12 refused with a reason (10 of
them 9B, one a misfiled LoRA), instead of one long load ending in a stack trace.

## 1.18.2 — First edit LoRA in the catalogue, and two traps it exposed

`Consistence-Edit` ([lrzjason/Consistance_Edit_Lora](https://huggingface.co/lrzjason/Consistance_Edit_Lora),
Apache-2.0, rank 128, ~368 MB) is the first edit preset that actually loads on
FLUX.2-klein-4B. It restores high-frequency detail while preserving colour and
structure — a finishing pass, not creative editing. Suggested weight 0.6.

Integrating it surfaced two bugs that would have hit any LoRA:

- **Mixed key dialects.** The file carries 160 PEFT keys (`lora_A`/`lora_B`) plus 40
  in `lora.down`/`lora.up`. peft loads what it recognises and creates a FRESH adapter
  for the rest, so the LoRA applied **partially and silently** — only a RuntimeWarning
  buried in the output. `_lora_needs_normalizing()` now detects this from the header
  and `_load_lora_normalized()` maps `down`→`A`, `up`→`B` before loading, and logs it.
- **Adapter namespace collision.** Upstream, edit and base are two models, so two
  independent adapter sets. Here they are the SAME object: both used `cz_lora_i`, so a
  base LoRA plus an edit preset died on "Adapter name cz_lora_0 already in use" — and
  `set_adapters` replacing the active list meant applying the edit set silently
  deactivated the base LoRAs. `_apply_edit_loras` now syncs the UNION of both sets
  (deduped by path, first weight wins) with a single source of truth.

Covered by tests/test_lora_dialect.py; verified on GPU with a real edit.

## 1.18.1 — Finish the rename, and three things it uncovered

The 1.18.0 port left the engine correct but the product still called itself
crispz-qwen-edit in places. Renaming it surfaced real bugs, not just labels:

- **`install.sh` still checked for `QwenImageImg2ImgPipeline`.** Only the Windows
  installer had been updated, so a Linux/macOS install ended on a false error.
- **Every GGUF FLUX.2 checkpoint was silently hidden.** `gguf_arch` was pinned to
  `qwen_image` in the config, and a FLUX.2 GGUF declares `flux`/`flux2`, so
  `list_checkpoints()` skipped it with "this build only loads 'qwen_image'". The
  accepted set is now `flux2,flux`, and because that label does NOT distinguish
  FLUX.1 from FLUX.2, the tensor **layout** is the real judge: the positive
  signature is now `x_embedder` + `context_embedder` +
  `double_stream_modulation_img`, so a FLUX.1 city96 GGUF is still rejected.
- **`preferences.example.json` and `presets/example.json`** shipped
  `Tongyi-MAI/Z-Image-Turbo`, so copying either gave an unloadable config.

Also rewritten for klein: `run_quality_rtx5090.bat` (it forced `CZ_OFFLOAD=model`
and an HF-offline note about a 46 GB Qwen cache — klein is 15 GB and wants
offload `none`), `fooocus_extra.json`, `config_modification_tutorial.txt`, the
launcher scripts, and the `_edit_loras_help` / `_gguf_arch_help` config strings.
`tools/check_zimage_models.py` watched for a Z-Image Omni/Edit release that klein
does not need; it becomes `tools/check_klein_extras.py`, which watches for what
klein actually lacks — a FLUX.2 ControlNet or edit LoRA.

## 1.18.0 — Engine: FLUX.2 Klein 4B, one model for the whole surface

Fork of crispz-qwen-edit. `Flux2KleinPipeline` does txt2img **and**
multi-reference editing (its `image` argument takes a list of PIL images), and
`Flux2KleinInpaintPipeline` does inpaint **and** img2img. So the four kinds of
`get_pipe` collapse onto **one loaded model**: 14.9 GB total on an RTX 5090
against two 20B models upstream, 1024x1024 in 2.0 s at 4 steps, an edit with one
reference in 3.0 s.

`Flux2KleinPipeline` exposes no `strength` - its `image` is a reference
conditioning, not a noised start. img2img therefore routes to the inpaint
pipeline with a **full white mask**, injected inside `_qwen_call` so that
`_refine_whole`, `_refine_tiled`, `process_one` and `txt2img_run(upscale=True)`
are untouched. `get_pipe("img2img")` and `get_pipe("inpaint")` return the same
object.

**No CFG and no negative prompt, and the app says so.** klein-4B is step-wise
distilled: renders at guidance 1.0 / 4.0 / 8.0 are bit-identical (MAE 0.0000,
`tests/test_klein_guidance.py`), and diffusers warns about it itself. `_cfg()`
returns `{}`, `_qwen_call` pins `guidance_scale=1.0` when the transformer reports
`is_distilled` (which also silences that per-call warning), and the CLI protocol
announces `supports.negative: false` plus a warning on any spec carrying a
`negative` or a `guidance`. The UI slider survives for API compatibility and
changes nothing.

The 19 Qwen-Image-Edit task LoRAs added in 1.17.0 and the 2 Lightning speed
presets are **removed from the catalogue**: they cannot load on FLUX.2, and
`caps` was advertising them. The upstream catalogue is kept commented in
`cz_edit_loras.py` as a merge reference; the lazy-download machinery is intact
and wakes up as soon as a FLUX.2 edit LoRA is declared. klein is already
distilled to 4 steps, so there is nothing left to accelerate: `edit_fast` is
`["off"]`.

The single-file / GGUF architecture guard looked for `img_in`, `txt_in`,
`time_text_embed` - none of which exist in a FLUX.2 transformer. Markers
re-derived from the real 169-tensor checkpoint: `single_transformer_blocks.`,
`double_stream_modulation`, `x_embedder`, `context_embedder`. A 7.2 GB
single-file `.safetensors` loads and renders.

## 1.17.0 — Edit LoRA presets: task LoRAs on the edit pipe, fetched on first use

The Reference (Omni) tab and the protocol op `edit` run on a SEPARATE pipe
(Qwen-Image-Edit), and the LoRA slots only ever reached the base pipe:
`_apply_loras` was called from `_ensure_base` / `_swap_transformer`, never from
`_load_omni`. So `spec.loras` on an edit, or a LoRA slot while editing, was
silently ignored - the edit came back without it and nothing said so.

Now the edit pipe has its own LoRA set (`EDIT_LORAS`, `set_edit_loras`,
`_apply_edit_loras`), hot-swapped in `generate_omni` through the same PEFT
sync as the base (`_sync_adapters`, shared code). A failure there RAISES
instead of degrading: the user asked for that LoRA, an edit without it is a
wrong result, not a fallback.

On top of it, the 19 task LoRAs of
[Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load)
become presets (`cz_edit_loras.py`): Photo-to-Anime, Any-Light, Light-Migration,
Upscaler (2K), Multiple-Angles, Style-Transfer, Polaroid, Pixar-3D, Noir comic,
Studio-DeLight... Lazy: nothing is fetched at import; the first selection
downloads the file from Hugging Face into `<loras_dir>/_hf-edit/<adapter>.safetensors`
(ASCII name - two upstream files are named in Chinese), after which it is an
ordinary LoRA file every existing path already understands. `config.txt
edit_loras` adds/replaces/removes presets, `edit_loras_dir` moves the folder.

- UI: a dropdown under the reference images (Reference (Omni) tab) lists the
  presets (`✓` on disk, `⬇` to fetch), with a weight slider and a "Use example
  prompt" button (the presets have no trigger word: the instruction is the
  prompt). A checkbox "Edit LoRAs" in Models > LoRA switches the whole edit set
  on/off without losing the selection - a one-click with/without comparison.
- Protocol (v1, additive): `spec.loras` on op `edit` accepts a preset name
  (`"Photo-to-Anime"`, `"upscale-2k:1.0"`) or a file, and lands on the edit
  set, never on the base. `caps.edit_loras` lists the presets (name, prompt,
  inputs, downloaded) and `supports.edit_presets` announces them. `guidance`
  is now APPLIED on `edit` (a distilled Lightning/Rapid setup needs 1.0 while
  the global slider stays at 4); on `gen` it keeps the v1 warning. An explicit
  `width`/`height` on `edit` is passed to the pipe (`size_explicit`), so the
  Upscaler preset can produce a 2x output; without it the edit keeps the
  input size as before.

Fast edit mode and shared LoRA libraries, same release:

- **Edit speed** (dropdown under the edit presets): `Lightning 4 steps` /
  `Lightning 8 steps` stack the lightx2v Lightning edit LoRA on the edit pipe
  (the 2509 or 2511 file is picked from the edit model name, looked up in the
  LoRA folders first, downloaded otherwise) and force N steps + CFG off;
  `Auto (model profile)` takes steps/guidance from `model_profiles` for an
  already-distilled edit model (Rapid-AIO, new `aio` profile = 4 steps,
  guidance 1.0); `Off` = Settings. The speed LoRA stacks AFTER the task
  presets and is independent of the "Edit LoRAs" checkbox. Protocol:
  `spec.fast` (`off|auto|lightning-4|lightning-8`) on op `edit`,
  `caps.edit_fast`; an explicit `spec.steps` is never overridden.
- **Rapid-AIO as the edit model**: `zimage_omni_model` now accepts any
  single-file transformer, `.gguf` OR `.safetensors` (FP8/INT8 scaled ComfyUI
  builds go through the same dequant + disk cache as the base). `_load_transformer`
  takes `(path, base)` so the edit pipe reuses the base loader instead of its own
  GGUF-only branch; `2511` repos select the Plus pipeline like `2509`.
- **`loras_extra_dirs`** (config, env `LORAS_EXTRA_DIRS`, prefs, UI textbox
  under the LoRA folder, CLI `--loras-extra-dir`): extra LoRA folders merged
  into every list (slots, XYZ, protocol `caps.loras`); on a duplicate name the
  main folder wins. `resolve_lora_path` searches them in order. Edit presets
  reuse a file already there (Civitai name for Multiple-Angles) instead of
  downloading a second copy.

Tests: `tests/test_edit_loras.py` (registry, lazy download under an ASCII
name, separate set, hot-swap + checkbox, failure raises, protocol routing,
caps, extra dirs merge + protocol listing, preset reuse from a library,
Lightning revision pick + stacking + Auto profile, `fast` aliases),
`test_protocol_edit.py` updated for the new `generate_omni` kwargs.

## Unreleased — GGUF: the tensor layout outranks the declared architecture

`general.architecture` is just a KV string, and conversion tools stamp it wrong:
`chapel/hyphoria_qwen_v1.0` ships Qwen-Image GGUFs (1933 tensors, pure diffusers
layout) declaring `wan`. The startup filter compared that string to `gguf_arch`
and skipped them - 17 GB of a perfectly loadable model never reaching the
dropdown, with a message blaming the wrong thing. The tensor NAMES are proof;
the label is a hint. So the layout now decides:

- Qwen-Image signature present -> accepted whatever the label says (a console
  line reports the mismatch instead of hiding it);
- foreign layout (stable-diffusion.cpp: `blocks.N.attn.wq`, `txtmlp`) ->
  rejected as before, even when it declares `qwen_image` - still the real guard,
  and still what catches `realismByStableYogi_v15TurboGGUF`;
- layout undetermined -> the declared architecture keeps the last word.

The existing `_GGUF_OK_PREFIXES` could NOT carry this: `transformer_blocks.`,
`time_text_embed`, `norm_out` and `proj_out` are shared with FLUX, so trusting
them over the label would have let a FLUX GGUF through. The new signature is
`img_in.weight` + `txt_in.weight` + `txt_norm.weight`, which FLUX does not have
(it uses `x_embedder`/`context_embedder`). Verified against the real files: both
hyphoria GGUFs carry all three; the 7 local GGUFs are unchanged.

Tests: `test_gguf_layout_beats_a_mislabelled_architecture`,
`test_gguf_foreign_layout_still_rejected`,
`test_gguf_flux_diffusers_layout_is_not_mistaken_for_qwen`,
`test_list_checkpoints_accepts_the_mislabelled_gguf`.

## Unreleased — the same fix on the non-quantized path + the crispz-krea2 loader methods

Follow-up to the ComfyUI-prefix fix below, which only covered FP8/INT8 files.

- **Non-quantized single-files too.** A plain bf16/fp16 `.safetensors` at the
  ComfyUI layout hit the identical `Cannot copy out of meta tensor` crash: it
  went straight to `from_single_file(path)`, and diffusers' identity mapping for
  `QwenImageTransformer2DModel` never strips `model.diffusion_model.`. The layout
  is now detected from the HEADER (`_safetensors_comfy_prefixed`, a few KB read)
  and such files are read + un-prefixed before loading. No extra RAM cost -
  `from_single_file` reads the whole checkpoint anyway - and NO dequant-cache
  entry is written for them (nothing was dequantized; it would be a bf16 -> bf16
  copy).
- **Prefix stripped at READ time** (crispz-krea2 `_read_comfy_state_dict`
  method), not on the way out: the arch guard, the weight-scale lookups and the
  `comfy_quant` blobs all work on diffusers-layout keys now, and
  `_quantization_metadata` layers are normalized once instead of being stored
  under two variants.
- **Dequant math on the GPU** (also from crispz-krea2): the cast + descale +
  un-rotation is memory-bandwidth-bound on CPU (~9 min measured there on a 12.9B
  INT8). It now runs tensor by tensor on the GPU when there is one - a few
  hundred MB of VRAM at peak - and the state dict still comes back in RAM. New
  config `convert_device`: `auto` (default) | `cpu`.

Tests: `test_bf16_prefixed_is_detected_and_stripped`,
`test_prefixed_single_file_is_loaded_as_a_state_dict` (proves the loader gets a
state dict, not a path), `test_convert_device_cpu_is_respected`,
`test_dequant_result_always_lands_on_cpu`.

## Unreleased — FP8/INT8 checkpoints: strip the ComfyUI key prefix

Every single-file Qwen checkpoint distributed by Comfy-Org / Civitai names its
tensors `model.diffusion_model.*`. diffusers 0.39 registers
`QwenImageTransformer2DModel` with an IDENTITY mapping function - it does no key
conversion at all - so that prefix has to be gone before the state dict reaches
`from_single_file`. The dequant loader filtered ON the prefix but never removed
it: all 1933 keys landed as 'unexpected', not one weight was loaded, the model
stayed on the 'meta' device and `dispatch_model` died with

    NotImplementedError: Cannot copy out of meta tensor; no data!

which read like a broken/incompatible checkpoint but hit EVERY FP8/INT8 Qwen
single-file (qwen_image_edit_2509_fp8_e4m3fn, jibMixQwen_v60, ...). The prefix
is now stripped on the way out of `_load_dequant_state_dict` (the internal
lookups - weight scales, comfy_quant blobs - keep working on the raw keys).
Verified against the real Edit-2509 FP8 file: stripped keys match
`Qwen-Image`'s transformer state dict exactly, 1933/1933, no missing, no extra,
no shape mismatch. Dequant-cache key bumped to `bf16-v2` so entries written
before this fix (prefixed keys, unloadable) are never reused. Tests:
`test_comfy_prefix_is_stripped` + `test_unprefixed_keys_left_alone`.

## Unreleased — prompt & negative boxes: capped growth + a visible scrollbar

A long prompt used to grow the textarea unpredictably (Gradio-version
dependent) and then CLIP silently, the text continuing below the fold with no
scroll cue on the dark theme. The prompt now grows to 12 lines (~+20%) and the
negative to 6, then SCROLLS - with a themed, visible scrollbar (CSS capped at
17em/9em as a version-proof backstop). Same behaviour across the whole family.

## Unreleased — face & hand detailers brought up to crispz-studio level

The family carried a PRE-FIX copy of the face detailer and no hand detailer at
all. Ported from crispz-studio, current version:

- **🖐 Hand detailer**: same circuit as faces (enlarged crop -> high-res
  img2img refine -> feathered paste), YOLOv8 hand detection running as a PURE
  ONNX session (onnxruntime, CPU by default). The torch/ultralytics model is
  exported once to cache/ in a SUBPROCESS and never imported in the app
  process - that isolation is a hard requirement: a merely-resident torch YOLO
  model corrupts the shared diffusion weights during offload transfers
  (checksum-proven on crispz-studio, 2026-08-17; renders degrade to mosaic
  then NaN). Config hand_detailer* keys; optional 'ultralytics' + 'onnx'
  packages (requirements-extra.txt) for the one-time export.
- **Local crop prompts** (face_detailer_prompt / hand_detailer_prompt, empty
  by default): feeding the SCENE prompt to a crop makes the model repaint the
  scene inside it (observed: a sign's text written on refined cheeks, a tiny
  face between thumb and index). Empty = the refine only sharpens the crop.
- UI: 🖐 checkbox + hand denoise slider (live), wired next to the face ones.

## Unreleased — dequant disk cache (ported from crispz-studio) + metadata-ConvRot fix

FP8/INT8 'scaled' checkpoints were re-dequantized at EVERY load (minutes on a
HDD). The bf16 result is now written once to cache/dequant (config
dequant_cache, LRU cap dequant_cache_max_gb) and the next loads of the same
checkpoint become a normal single-file read (seconds). Keyed on the ORIGINAL
file (path+size+mtime) - deleting the cache is always safe, it rebuilds on
demand. Also ported: comfy-quants can declare the int8_tensorwise/ConvRot
scheme centrally in __metadata__._quantization_metadata instead of per-tensor
blobs; ignoring that variant dequantized such checkpoints to pure noise
(caught on Krea 2, same format possible here). Both styles are now read.
Tests: tests/test_dequant_cache.py (5) + metadata-convrot roundtrip.

## Unreleased — queue: persistence + soft ⏸ Pause (ported from crispz-studio)

The job queue now SURVIVES a restart or crash: saved to cache/queue.json after
every mutation (add/move/remove/clear) AND after every job, restored at startup
(the accordion shows the restored count; job_queue.persist=false to disable).
Images in job snapshots are written under cache/queue_assets/. Two loss-free
ways to halt a run: the new ⏸ Pause button finishes the current job then
suspends; Stop interrupts mid-render and the interrupted job now STAYS queued
(it was discarded before) - it re-runs entirely on resume.

## Unreleased — AI provenance: C2PA reading + TrustMark invisible watermark (EU AI Act art. 50)

New optional brick `cz_provenance.py` (CPU only, the GPU is never touched), for
machine-readable AI disclosure as required by EU AI Act Article 50 (applicable
Aug 2 2026; systems already on the market have until Dec 2 2026):

- **Read** (PNG Info + CLI `--provenance -i img.png`): a **Provenance** section shows
  any embedded **C2PA / Content Credentials** manifest (issuer, claim generator,
  signature state — Firefly/ChatGPT/Gemini outputs carry these), automatically and for
  free; the **🔍 Check invisible watermark** button decodes a **TrustMark** watermark
  on demand (lazy model init ~4 s once, then ~0.1 s/image). The UI wording is
  deliberate: *absence of marks proves nothing* — it never claims "not AI"/"authentic".
- **Write** (`save_image()`, the single choke point all saves go through — txt2img,
  upscale, inpaint, reframe, queue, CLI, HTTP endpoints): with
  `provenance_watermark: "on"` every saved image gets an invisible TrustMark watermark
  carrying `provenance_wm_id` (**max 9 ASCII chars** — the error-corrected payload is
  ~68 bits; longer ids are truncated). Verified by test to survive PNG **and JPEG q95**
  re-encoding. Default **off**.
- Deps: `trustmark` + `c2pa-python` in `requirements-extra.txt` (graceful degradation
  when absent, same pattern as rembg). Windows note: install trustmark with
  `PYTHONUTF8=1` (its sdist crashes on cp1252). Tests: `tests/test_provenance.py`
  (7 cases, skip-if-missing so CI stays light). C2PA *writing* (signed manifests) is
  deliberately not included yet — it needs a signing-certificate decision first.
- Caution: TrustMark loads a small torch model **in-process** (CPU). Watch the first renders with
  `provenance_watermark: on` + GGUF/offload `model`; the watermark hook runs at save
  time only, and stays off by default.

## Unreleased — XYZ grid: full-Prompt A/B axis + type-ahead suggestions

**Why.** Comparing whole prompts needed Prompt S/R gymnastics, and filling the
Checkpoint/LoRA value fields meant copy-pasting long file names by hand.

**What.**
- **New `Prompt` axis**: each value is a **complete prompt** (quotes protect embedded
  commas) — true A/B/C testing, combinable with any other axis (e.g. Prompt ×
  Checkpoint). CLI too: `--xyz "Prompt=a, \"b, with comma\", c"`, and `--prompt`
  becomes optional when a Prompt axis is given. Sheet/job labels truncate long prompts.
  NB: the case-insensitive shorthand `prompt` now resolves to this axis (exact match
  wins); `prompt s` still reaches `Prompt S/R`.
- **Type-ahead in the X/Y/Z value fields** (after 3 typed characters, ↑/↓ + Tab/Enter,
  Escape, click): suggests from the **checkpoint/LoRA lists validated at startup** on
  the `Checkpoint` / `LoRA` / `LoRA + weight` axes (`:1` auto-appended for the latter),
  and from the local **`__wildcards__`** on the `Prompt` / `Prompt S/R` axes when the
  current token starts with `__`. CSV segments are respected (completion only touches
  the segment being typed). Disable with `xyz_grid.suggest: false` (same key as the
  ⤵ suggest button). Validated live in the browser: checkpoint filter + insert, second
  segment after a comma, wildcard expansion inside a prompt, `:1` suffix.

## Unreleased — Fix: ConvRot INT8 checkpoints rendered pure noise

Ported from crispz-studio: the dequant loader now parses the `comfy_quant` JSON blobs
and un-rotates ConvRot weights (grouped Hadamard, comfy-quants' specific H4-Kronecker
matrix — NOT Sylvester) after descaling. Without it, `int8_tensorwise` + `convrot`
checkpoints load structurally but render pure noise. Roundtrip unit test added.

## Unreleased — CLI parity: expand / inpaint-mask / reframe-fit / force-ratio flags

Ported from crispz-studio: **`--expand left,right,top,bottom`** (+ `--expand-ratio`) =
the "Expand sides" directional outpaint; **`--inpaint-mask mask.png`**
(+ `--inpaint-denoise`) = mask-file inpaint; **`--reframe-fit contain|cover`**;
**`--force-ratio W:H`** + **`--force-ratio-mode crop|extend`** (the `-o` single-file
path now honours the forced ratio like the standard path). Documented in README_CLI.md;
`--reframe … --reframe-fit cover` smoke-tested (1376×768, no fill).

## Unreleased — Force aspect ratio: new "Extend (outpaint)" mode next to crop

Ported from crispz-studio: the Upscale/img2img "Force aspect ratio" checkbox becomes a
radio — **Off / Crop to fit / Extend (outpaint)**. Extend reaches the target ratio by
outpainting the missing bands (symmetric, `outpaint_directions`: full-res centre kept,
~1 MP diffusion recomposed) instead of cropping, then a light **seam-blend pass**
(img2img denoise `force_ratio_extend_denoise`, default 0.22, 0 = off) is composited
back **only over the bands + a feathered ~5% transition margin** — the original centre
stays pixel-for-pixel untouched. Config: `force_ratio_mode` (`crop`/`extend`), env
`CZ_FORCE_RATIO_MODE`. Unit tests: `tests/test_force_ratio.py`.

## Unreleased — FP8/INT8 "scaled" checkpoints loadable (dequantized at load)

**Why.** Ported from crispz-studio: most CivitAI/HF Qwen-Image builds ship as ComfyUI
**FP8/INT8 "scaled"** safetensors (e.g. `qwen_image_edit_2509_fp8_e4m3fn`, 19 GB vs
38 GB bf16) and were skipped from the model list. GGUF was already supported here.

**What.** ComfyUI-quantized safetensors (`X.weight` F8/I8 + `X.weight_scale` scalar or
per-row + `X.comfy_quant` blob) are **dequantized in RAM to BF16** at load, then fed to
`from_single_file` as a state dict (diffusers key conversion still applies). AIO bundles
are filtered to `model.diffusion_model.*` (base VAE/text encoder reused). Tensors are
read in file-offset order (HDD-friendly). Still refused with clear messages: misfiled
LoRAs, SVDQuant/Nunchaku INT4, quantized checkpoints of a different architecture. Note:
a dequantized FP8 has the RAM/VRAM footprint of the full BF16 build (~38 GB for
Qwen-Image) — on this machine's 64 GB the GGUF builds remain the practical choice; the
FP8 path mainly unlocks fine-tunes that ship in no other format. Validated: real-file
detection (`fp8_e4m3fn` and `jibMixQwen` → dequant path, SVDQuant still rejected),
synthetic dequant math + AIO filtering + guards (`tests/test_quant_formats.py`, 8 cases).

## Unreleased — Fix: a LoRA picked as checkpoint no longer hunts for an SD1.5 config

**Why.** A LoRA file misfiled in a checkpoints folder (e.g. `ZITnsfwLoRAv3.safetensors`)
could be selected as the transformer: diffusers cannot recognise the state dict, falls
back to its default single-file repo and dies with
`OSError: stable-diffusion-v1-5/stable-diffusion-v1-5 does not appear to have a file
named config.json` — observed on a Pinokio install.

**What.** `_safetensors_unsupported` now detects LoRA state dicts from the header
(kohya `lora_down/up` + `lora_unet_/lora_te` prefixes, peft `lora_A/B`): such files are
skipped from the checkpoint list and force-loading one raises "LoRA file, not a
checkpoint - move it to the LoRA folder and pick it in Models > LoRA". The Z-Image
`from_single_file` also passes `config=BASE_REPO, subfolder="transformer"` (as the Qwen
fork already did), so a valid-but-unrecognised transformer never falls back to SD1.5 and
offline mode keeps working. Validated: 3 real Z-Image LoRAs detected, a real checkpoint
accepted and actually loaded through the new path (25 s, valid model), the forced-LoRA
guard raises the clear error, smoke 22/22.

## Unreleased — Fix: concurrent generations no longer corrupt the shared scheduler

**Why.** Gradio does not serialise events from DIFFERENT listeners: a manual **Generate**
still running while **Run queue** starts its first job (or the face detailer refining)
meant two threads calling the same shared pipeline and stepping the SAME scheduler — its
internal index ran past the end and the job died with
`IndexError: index 31 is out of bounds for dimension 0 with size 31`
(`scheduling_flow_match_euler_discrete.step`), observed live on crispz-qwen-edit.

**What.** A process-wide **GPU lock** (`_GPU_LOCK`, re-entrant) now serialises every
generation entry point (`generate`, `txt2img_run`, `process_one`, `_refine_whole`,
`outpaint`, `inpaint_run`, `generate_omni` — decorator `@_gpu_serial`): a second request
simply waits for the GPU instead of racing it. Same-thread nesting (txt2img→generate,
process_one→refine, detailer) stays free thanks to the RLock. Validated: 4 threads on a
locked function show zero overlap, nesting does not deadlock, all entry points wrapped,
smoke 22/22.
## Unreleased — GGUF: reject non-standard tensor layouts with a clear message

**Why.** Some Civitai GGUFs (e.g. `realismByStableYogi_v15TurboGGUF.gguf`) declare
`general.architecture=qwen_image` but were converted with a compact renamed tensor
scheme (`blocks.0.attn.wq`, `txtmlp`, `tproj`… — a stable-diffusion.cpp-style layout).
diffusers' GGUF loader can only map the ORIGINAL Qwen key names (QuantStack/city96-style
GGUFs): with such a file zero keys match, every weight stays on the `meta` device and
generation crashes with the cryptic *"Cannot copy out of meta tensor; no data!"*.

**What.** New `_gguf_layout_unsupported()` header check (reads tensor names only, no
weights): such files are now **skipped from the checkpoint list** with an explicit log
line, and force-loading one raises a clear error naming the file and the fix (use a
QuantStack/city96 GGUF or the BF16/FP16 `.safetensors` build). Validated on the real
file (rejected) and the three standard-layout GGUFs (Edit-2509, Rapid-AIO, qwen-image —
still listed); smoke 22/22.

## Unreleased — Thumbnail cache: app-folder default, UI field, and CLI flags for the new features

- **New default location**: the Asset Browser thumbnail cache now lives in **`<app>/cache/`**
  (gitignored) instead of inside the output folder — the app folder is usually on a fast
  disk, so the grid stays fast even when outputs are on an HDD/NAS, with zero configuration.
  The special value **`output`** restores the old next-to-the-images layout
  (`<out>/_index/thumbs`, relative URLs); any path still works as a custom cache.
- **UI field**: *Save > Asset Browser > Thumbnail cache folder* (+ Save) — persisted to
  `preferences.json` (which now overrides `asset_browser.cache_dir` from config), applied
  immediately for writing; a restart is needed to *serve* from a brand-new path
  (`allowed_paths` is fixed at launch).
- **CLI**: the recent features are usable headless too — **`--detail-faces`** (+
  `--detailer-denoise`) runs the auto face detailer after a `--txt2img` render, and
  **`--metadata-scheme a1111`** writes the Civitai-readable `parameters` chunk. `--auth`
  was already in. Validated: default resolves to the app cache (the 10 951 migrated
  thumbnails are picked up unchanged — same output-dir slug), `output`/custom overrides,
  the UI handler, `--help`, build_ui, smoke 22/22.

## 1.16.0 — 2026-08-04 — Release: Fooocus-parity pass (auth, CivitAI consensus, face detailer) + SSD thumb cache

Consolidates everything since **1.15.0** — the gap-analysis pass against Fooocus2026:
optional **login page** (auth) for LAN/tunnel exposure, **CivitAI community recommended
settings** with one-click apply, **PNG Info ✨ Apply all**, one-click **🎲 Vary
(subtle/strong)**, the **🔧 auto face detailer** (ADetailer-style), the Asset Browser
**thumbnail cache on a fast disk** (`asset_browser.cache_dir`), **17 exact aspect
ratios**, the `simple` schedule alias, the same-base-model **⚠ update badge** fix, and
**occlusion-aware Face Swap** blending. Details in the sections below.

### 🔧 Auto face detailer (ADetailer-style)

**Why (last real Fooocus-parity gap).** Small faces in a wide shot come out soft: the
model spends ~1 Mpix on the whole scene, so a 150 px face gets almost none of it.
Fooocus's *Enhance* / A1111's *ADetailer* fix this by re-rendering each face at high
resolution; crispz had nothing equivalent.

**What.** New `cz_detailer.py` + a **🔧 Detail faces** checkbox under the Generate button
(module flag — the queue and X/Y/Z grid snapshots are untouched). After each render (and
after the optional auto-upscale), faces are detected with insightface buffalo_l (already
loaded for Face Swap; detection no longer requires the inswapper model —
`cz_face.detect_faces` / `_ensure_face_detector` factored out) and each face, biggest
first (up to `face_detailer_max_faces`, default 4), is:
enlarged crop (+60 % context) → scaled to the model's ~832 px sweet spot → **Z-Image
img2img** (same prompt/seed, `face_detailer_denoise` 0.35, live slider in Advanced >
Generation) → scaled back → pasted through a **feathered elliptical mask** (no square
edges, same technique as the Face Swap GFPGAN paste). Full-frame portraits are skipped
(nothing to gain), failures degrade to the untouched image.
Validated: geometry/mask units, build_ui, smoke 22/22 — then **end-to-end on a real GPU
render** (832×1216 classical-portrait scene, seed 20260804): txt2img 4.5 s, one face
detected, refined at denoise 0.35 / 12 steps in ~30 s (one-time insightface load +
img2img-pipe derivation included; subsequent images are much faster). Pixel-diff check:
**4.62 mean inside the face crop, 0.0000 outside** — the feathered paste is surgical,
the rest of the image is untouched to the pixel. Visually: a ring-shaped artifact on the
forehead ornament became a clean dot, skin gradients and lashes tightened, identity fully
preserved, no visible seam.

### PNG Info "✨ Apply all" + one-click "🎲 Vary" (Fooocus parity)

- **PNG Info** could only send the prompt and the seed. The new **✨ Apply all** button
  loads *everything* the image carries — prompt, negative, seed, steps, CFG, size
  (width/height), sampler/schedule — like Fooocus's full parameter load. crispz's own
  `sampler/schedule` notation is applied as-is (aliases like `simple` normalised); A1111/
  CivitAI sampler names go through the same conservative mapping as the CivitAI
  recommended settings (`Euler a` → euler, `DPM++ 2M Karras` → keeps the sampler, applies
  the karras schedule) and the status line spells out what was applied vs kept.
- **Vary (subtle / strong)** — two buttons in the Upscale/img2img tab arm a pure img2img
  pass in one click (ESRGAN off, refine on, denoise **0.25** / **0.6**), tick *Input
  Image* and open its panel; drop an image and press Generate. The report line explains
  what was armed.

### Security: optional login page (`auth` / `--auth` / `CRISPZ_AUTH`)

**Why.** The UI can be exposed on a LAN or through a tunnel (cloudflared) — until now with
no protection at all: anyone with the URL could generate images and browse/delete outputs.

**What.** Optional auth, off by default (localhost unchanged). Set config `auth` to
`"user:password"` (several accounts via commas), or pass `--auth user:pw`, or the
`CRISPZ_AUTH` env var: Gradio then shows a login page and gates every route — verified
live: without login, `file=` serving and API endpoints return 401; a wrong password is
rejected; a no-auth launch behaves exactly as before.

### CivitAI: community "recommended settings" (consensus) + one-click apply

**Why (Fooocus2026 parity).** CivitAI's example images publish their generation `meta`
(sampler, cfgScale, steps, size). Fooocus2026 analyses them into consensus settings;
crispz fetched previews/triggers/examples but ignored the settings.

**What.** `cz_civitai.analyze_settings` computes the consensus (median steps/CFG, majority
sampler/size, with the number of images used) and stores it as `"recommended"` in
`<name>.civitai.json` on every fetch. The Asset Browser model card shows a **Community
settings** block. In **Models > Checkpoints**, a new **📊 Apply CivitAI recommended
settings** button fetches (with progress — hashing a 12 GB checkpoint without a cached
hash takes minutes on an HDD) and applies steps/CFG plus sampler/schedule when a Z-Image
equivalent exists (`Euler*` → euler, `UniPC` → unipc, `LCM` → lcm, `*Karras*` → karras
schedule…); DPM++-style samplers with no equivalent are reported and left unchanged.
Validated against live CivitAI: consensus `{steps 9, CFG 1.0, sampler Euler}` from 5
community images, applied as steps=9 / CFG=1.0 / euler.

### Asset Browser: thumbnail cache on a fast disk (`asset_browser.cache_dir`)

**Why.** Thumbnails are the Asset Browser's hot path — one file per image, re-read on
every grid paint — and they were always written next to the images. With the output
folder on a slow HDD (plus antivirus write-scanning), serving a single 84 KB thumbnail
cold took **~1.1 s**; a grid of thousands crawled.

**What.** New `asset_browser.cache_dir` (config, empty by default). When set to a fast
disk (e.g. `"D:/crispz-cache"`), thumbnails go to `<cache_dir>/crispz-thumbs/<slug>/`
(one slug per output folder — several output folders never collide) and the manifests
reference them by absolute `/gradio_api/file=` URL; the launcher serves that folder
automatically. Empty keeps the previous `<out>/_index/thumbs` layout, so nothing changes
for existing setups. The path logic is centralised (`_thumbs_root` / `_thumb_paths`) and
used everywhere thumbs are built, checked, or deleted — `delete_asset` and freshness
checks follow the cache. The cache is disposable: delete it any time, it rebuilds on
demand. Measured on a 10 576-image folder (HDD → SSD): cold thumbnail ~1.1 s → **~3 ms**,
30 grid thumbnails in **88 ms** total.

### Aspect ratio: 8 sizes to 17, sorted, with the exact CivitAI ratios

The dropdown only carried the Fooocus list, whose portrait/landscape entries
(`832 x 1216`, `768 x 1344`, `1536 x 640`) are *approximations* of 2:3, 9:16 and 21:9,
and which had no 5:4 or 4:3 at all. Following a CivitAI or ComfyUI recipe meant accepting
a different framing or dropping to the CLI (`--gen-width` / `--gen-height`).

Nine sizes added, all multiples of 16, all **exact** ratios except where they mirror an
existing Fooocus label: `1280 x 1024` / `1024 x 1280` (5:4, 4:5), `1280 x 960` /
`960 x 1280` (4:3, 3:4), `1536 x 1024` / `1024 x 1536` (3:2, 2:3), `1536 x 864` /
`864 x 1536` (16:9, 9:16), `640 x 1536` (the missing portrait ultra-wide). The Fooocus
entries stay as they are — existing presets and seeds depend on them.

The list is now sorted from squarest to widest, each landscape size followed by its
portrait counterpart, instead of the historical order. Cost runs 1,0 to 1,6 Mpix, so the
largest ones are slower — worth it when the recipe you are following calls for them.


### Schedule: `simple` accepted as an alias of `sgm_uniform`

**Why.** ComfyUI and CivitAI recipes name the native flow schedule `simple`. Ours was
only reachable as `sgm_uniform`, so every copied recipe needed a mental translation and
`--schedule simple` was rejected outright.

**What.** `simple` is now accepted wherever a schedule is written (`default_schedule`,
`ZIMAGE_SCHEDULE`, `--schedule`, the XYZ `Schedule` axis) and normalised back to
`sgm_uniform`, so metadata and presets keep one name for one curve. It is **not** a
second entry in the UI dropdown: same schedule, not a new option. The sigmas the pipeline
hands to the scheduler are `linspace(1, 1/steps, steps)` — exactly what ComfyUI's
`simple` produces on a flow model, so the alias is an equality, not an approximation.

### CivitAI: the ⚠ update badge now compares within the same base model

**The bug.** The check took `modelVersions[0]` from `GET /models/<id>` — the most recent
version of the *page*, whatever it was trained on. LoRA pages routinely move on to a
different base (a *Flux* or *Z-Image* version landing on the page of a *Qwen-Image-Edit* LoRA), and every local copy was then flagged as
outdated, pointing at a file that would not even load in this app.

**The fix.** `get_latest_version` takes the local `baseModel` (from the sidecar, or
deduced from our own version inside the same response for pre-existing sidecars) and only
considers versions published for that base — normalised comparison, so *Qwen-Image-Edit* / *qwenimageedit*
match. No version shares our base → no update, rather than a false positive. If the API
returns no `baseModel` at all, nothing is filtered: the information is missing, not
contradictory. Stale flags clear on the next `civitai_index` pass (or **🔄 Fetch all
missing**), which re-checks already-enriched models without re-downloading.

### Face Swap: occlusion-aware blending, and ONNX actually on the GPU

**The bug.** inswapper renders a 128 px face and insightface pastes it back through a
plain rectangle (`img_white` in `model_zoo/inswapper.py`), which is blind to depth: on a
shot of someone eating, the ice cream and the hand holding it sat inside that rectangle
and were repainted by the generated face. The mouth area came out broken on every image
where an object touches the face.

**The fix.** `_faceswap` now uses `paste_back=False` and composites itself, through a
mask built from an **occlusion pass** (XSeg, `dfl_xseg.onnx`) intersected with a
**face-region pass** (BiSeNet, `bisenet_resnet_34.onnx`), both computed on the *original*
frame — the only place the occluding object is still visible. The restore pass is masked
the same way, so it cannot repaint over an occlusion either. Added **LAB colour matching**
between the swapped and original face, and **CodeFormer** as the default enhancer
(`faceswap_restore_model`, with a `faceswap_restore_fidelity` weight) over GFPGAN.
`faceswap_restore` now defaults to **on**: without it a 128 px swap is visibly soft.
Each model is fetched once and every pass degrades cleanly to a log line if absent.
Cost: ~90 ms per face on GPU.

**GPU.** `requirements-lock.txt` pinned both `onnxruntime` and `onnxruntime-gpu` (rembg
pulls the CPU build). Same module name, so the CPU build **shadowed** the GPU one and
every ONNX pass — swap, restore, masks, rembg — had been running on CPU. The installers
now filter the CPU build out of the lock, like the existing Pillow filter.

**Not** a replacement for Fooocus-style FaceSwap, which conditions the diffusion itself
(IP-Adapter face) rather than pasting afterwards. No IP-Adapter or ControlNet exists for
Z-Image-Turbo today — only LoRAs — so the post-process remains the only route here.

## 1.15.0 — 2026-07-28 — Release: Asset Browser rearchitecture, security, GPU-agnostic tooling

Consolidates everything since **v1.11.2**. Nothing new here beyond the last build change
below — this entry marks the release boundary.

**Asset Browser — the big one.** Opening it re-read the PNG metadata of every image on
every open (**295 s** for 9 278 images) and shipped a **9,42 MB** manifest to the browser,
while the SPA stopped polling after 180 s — so it never finished filling in. Rebuilt on the
Fooocus design: a metadata cache (295 s → **3,7 s**), a tiny `days.json` plus one manifest
per day (**5 400× less data** on open), and incremental indexing at save time (~15 ms/image,
no rescan). Global search was kept, which Fooocus does not have on its Outputs tab.

**Security.** 37 Dependabot alerts triaged against what the code actually calls: Pillow,
protobuf and sentencepiece upgraded (**21 closed, 0 advisories left** on those pins), the
other 16 assessed unreachable and dismissed with per-package reasons — documented in
`SECURITY.md`, and the four Dependabot PRs closed with the reasoning.

**Tooling.** `boot_check.bat` replaces the RTX-5090-only script and works on any card; its
decisive check compares the GPU's `sm_XX` against the installed torch build, catching the
`WinError 127 torch_cuda.dll` class of failure *before* launch. `update.bat`/`.sh` add the
missing post-`git pull` step. New `lcm` sampler; LoRA weights can go negative (`-2..2`).

- Last change in this release: the `pillow==12.3.0` pin stays in `requirements-lock.txt`
  (so Dependabot sees the fixed version) and `install.*` / `update.*` filter that one line
  out before `pip` runs, installing Pillow separately with `--no-deps`. Commenting it out
  had fixed the install but left Dependabot matching the whole advisory range.

## 1.14.1 — 2026-07-27 — Fix install (Pillow/gradio), sampler status, drop the RTX-5090 scripts

Fallout from testing `install.bat` / `update.bat` / `run.bat` end to end.

- **`install.bat` was broken** by the 1.13.1 security bump: `gradio 5.50` declares
  `pillow<12.0`, so pinning `pillow==12.3.0` made a clean install fail with
  `ResolutionImpossible`. The running venv had not noticed because the upgrade used
  `--no-deps`. There is **no Pillow below 12 that fixes those CVEs** (11.3.0 is the last
  11.x and leaves 19 advisories open), so Pillow is now installed **separately, after the
  lock, with `--no-deps`** by `install.bat`/`.sh` — and re-applied by `update.bat`/`.sh` so
  an update cannot silently regress it. gradio's bound is conservative; Pillow 12 is
  verified working here (build_ui, save + metadata round-trip, thumbnails, `RankFilter`,
  `crop`, WebP). The `<6` gradio pin stays deliberate.
- **Sampler dropdown warning fixed**: `set_sampler` / `set_schedule` return a status
  string but were wired to `None` outputs, so Gradio logged *"returned too many output
  values"* on every change (any sampler, not just the new `lcm`). The status is now
  **displayed** next to the dropdowns instead of being discarded.
- **RTX-5090 scripts removed**: `boot_check_rtx5090.bat`, `run_quality_rtx5090.bat`,
  `_lan`, `_web`. They are superseded by `boot_check.bat` / `_lan` / `_web`, which are
  card-agnostic. Their one behaviour not yet covered — a fixed `GRADIO_SERVER_PORT=7860` —
  was carried over. No launcher (including the Pinokio one) referenced them.
- `config.txt` regains the inline `_help` documentation it had lost historically, plus the
  24 keys added since it was created — user values preserved.
- Files: `install.bat`, `install.sh`, `update.bat`, `update.sh`, `requirements-lock.txt`,
  `cz_ui.py`, `boot_check.bat`, `README.md`, `VALIDATION.md`.

## 1.14.0 — 2026-07-27 — Smart boot check (any GPU) + update scripts

`boot_check_rtx5090.bat` only knew one card and hardcoded a model path. Replaced by a
generic diagnostic, and the missing post-`git pull` step now exists.

- **`boot_check.bat`** — works on any NVIDIA card. Beyond driver/VRAM/temperature, it runs
  the check that actually matters: **is the GPU's `sm_XX` in the installed torch build's
  arch list?** That is precisely the *"RTX 50xx + non-cu128 torch"* failure
  (`WinError 127 … torch_cuda.dll`) hit earlier in this project — it is now caught **before
  launch**, with the exact `pip install --index-url …` line to fix it, and the script stops
  instead of letting the app die at the first CUDA allocation.
- **Recommendations scale with the card**: `_hw_check.py` now maps compute capability to a
  generation name (Blackwell / Ada / Ampere / Turing / Pascal) and derives CPU offload,
  ESRGAN tiling, max resolution and dtype from the real VRAM. Thresholds come from figures
  measured in this project (FLUX bf16 ≈ 33 GB with its encoder, GGUF Q8 ≈ 12,7 GB,
  `sequential` ≈ 3 s/step vs `model` ≈ 1,1 s/step). The 12 GB tier is keyed at 11 GB, since
  a "12 GB" card reports ~11,9 — putting it on `sequential` would have cost 3x for nothing.
- **Model folders are read from `config.txt`** instead of a hardcoded `D:\…\Z-Image`.
- **`boot_check_lan.bat` / `boot_check_web.bat`** — same diagnostic then LAN (`0.0.0.0`) or
  Cloudflare tunnel; both print a **no-authentication warning** first, consistent with
  `SECURITY.md`. `boot_check_rtx5090.bat` is kept as an alias.
- **`update.bat` / `update.sh`** — the post-GitHub-update step: refuses to `git pull` over
  uncommitted work (shows what is dirty), reinstalls dependencies **only if the requirements
  file changed** (md5), **warns if `torch` was replaced** (a transitive resolve can swap a
  `+cu128` build for a CPU wheel — that happened here once), re-runs the hardware check,
  verifies diffusers and `cz_ui` still import, and lists **new config keys** from
  `config-sample.txt` (`config.txt` is never overwritten). Flags: `--no-pull`,
  `--force-deps`, `--shared`.
- All `.bat` written with **CRLF**: `goto` silently fails on LF-only batch files.
- Files: `boot_check.bat`, `boot_check_lan.bat`, `boot_check_web.bat`,
  `boot_check_rtx5090.bat` (alias), `update.bat`, `update.sh`, `_hw_check.py`, `README.md`.

## 1.13.1 — 2026-07-27 — Security: upgrade Pillow / protobuf / sentencepiece (21 Dependabot alerts)

Adding `requirements-lock.txt` made Dependabot match **pinned versions** instead of ranges,
raising 37 alerts. Each was triaged against what the code actually calls.

- **Upgraded**: `pillow 11.3.0 -> 12.3.0`, `protobuf 6.31.0 -> 7.35.1`,
  `sentencepiece 0.1.96 -> 0.2.2` — **21 alerts closed**, and the advisory database reports
  **0 remaining** for those three pins.
- Pillow was the priority: it is the only flagged package that parses **files the user
  supplies** (Input image, PNG Info), so its 18 advisories (PSD/FITS/JPEG2000 OOB, font and
  PDF decompression bombs, `RankFilter`, `ImageCmsTransform`, `crop`/`paste` overflow…)
  were genuinely reachable.
- **Deliberately not upgraded** (documented per-package in `SECURITY.md`): rembg (server-only
  flaws, server never started, patched line needs Python 3.11 while this runs 3.10), gradio
  (`gr.load()`/OAuth/audio unused, Windows traversal needs Python 3.13+; fix needs a 6.x
  major bump the `<6` pin deliberately excludes), transformers (`Trainer`/LightGlue unused,
  `trust_remote_code` never set; fix needs 5.x), torch (`jit.script`/`lstm_cell`/
  `unpack_sequence` never called; fixes have no `+cu128` build, so upgrading would break
  RTX 5090 support to close unreachable flaws).
- `torch` was protected during the upgrade (`pip install --no-deps`) — the earlier incident
  where a transitive resolve replaced `2.8.0+cu128` with a CPU build must not repeat.
- Verified on the real environment: `torch 2.8.0+cu128` + CUDA still load, `build_ui()`
  builds, the image chain (save + metadata round-trip + thumbnail + `RankFilter` + `crop` +
  WebP) works under Pillow 12, and the test suites pass.
- `SECURITY.md`'s alert section was rewritten: it still claimed the repo had *no lockfile*,
  which is what changed.
- Files: `requirements-lock.txt`, `SECURITY.md`.

## 1.13.0 — 2026-07-27 — Asset Browser: per-day index + incremental indexing (Fooocus architecture)

Follow-up to 1.12.2. The metadata cache fixed the *server* side (295 s -> 3,7 s), but the
browser still downloaded and rendered a **9,42 MB manifest with all 9 278 images** on every
open. Aligned on the Fooocus design, which was studied for this.

- **`_index/days.json`** — a tiny index (`{date, count}` per day, ~200 bytes for 42 days).
  The page reads *that* on open, so the sidebar and the current day appear immediately
  instead of waiting for a multi-megabyte manifest.
- **One `manifest.json` per day**, written *inside* the day folder (Fooocus convention).
  The SPA loads only the day being displayed: **9,42 MB -> 1,48 MB** for the largest day
  (938 images), and typically far less. **5400x less data** for the initial load.
- **Incremental indexing** — new `on_image_saved()` hook (crispz's `on_image_logged`),
  called from `save_image()`: thumbnail + day manifest + `days.json` are updated **as the
  image is written** (~15 ms), so the browser no longer needs a folder rescan to be current.
  Idempotent (re-saving the same file does not duplicate it) and silent by contract — any
  failure is logged and *never* breaks a generation.
- **Global search preserved.** Fooocus only searches within the LoRA/Models tabs; crispz
  searches all output metadata, so that was kept: typing a query loads the remaining days
  in the background (cached in memory) and searches across everything.
- **Backwards compatible**: the global `_index/manifest.json` is still written, and the SPA
  falls back to it when `days.json` is absent (index not migrated yet).
- `_entry_for()` is now the single definition of a manifest entry, shared by the full
  reindex and the incremental hook, so the two paths cannot drift apart.
- Files: `cz_assetbrowser.py` (`_write_day_manifests`, `on_image_saved`, `_bump_days_index`,
  `_entry_for`, `_INCR_LOCK`), `cz_imageio.py` (hook in `save_image`), `cz_assets.py`
  (SPA: `days.json` -> per-day load, search loads all days), `tests/test_ab_index.py`
  (+5 tests: per-day manifests, incremental add, idempotence, never raises, identical
  entry shape between both paths).

## 1.12.2 — 2026-07-27 — Asset Browser: metadata cache (reindex 80x faster)

Opening the Asset Browser re-read the PNG metadata of **every** image on every open —
~25 ms each. Measured on a real 9 278-image library: **295 s per open**, while the SPA
gives up polling after 180 s (90 x 2 s). The manifest was therefore written *after* the
page stopped listening: "it doesn't refresh" and "images are missing".

- `ab_reindex` now keeps a **metadata cache** (`_index/meta_cache.json`, rel -> mtime+size
  signature + parsed meta). Unchanged files are served from it; only new or modified
  images are re-read.
- Measured on the same 9 278 images: **295 s -> 3,7 s (80x)**. Well under the polling
  window, so the gallery fills in as intended.
- Cache follows deletions (entries for vanished files are dropped, no unbounded growth)
  and is **defensive**: a corrupt/unreadable cache is ignored and rebuilt, never fatal.
- Each pass logs what it did: `indexed N image(s) in X.Xs (H from meta cache, R read)`.
- Files: `cz_assetbrowser.py` (`_load_meta_cache`, `_save_meta_cache`, `_meta_cached`),
  `tests/test_ab_index.py` (5 tests: cache hit, modified file re-read, fresh metadata
  reaches the manifest, deletions pruned, corrupt cache non-fatal).

## 1.12.1 — 2026-07-27 — New `lcm` sampler (LCM flow-matching)

- **Sampler** gains **`lcm`** (`FlowMatchLCMScheduler`) next to `euler` and `unipc`:
  designed for **few steps with guidance ~0-1**, so it suits distilled / Turbo checkpoints.
  Works with all four schedules (`sgm_uniform` / `beta` / `karras` / `exponential`) — the
  12 sampler×schedule combinations were verified to build.
- Falls back to `euler` (with a log line) if the installed diffusers has no
  `FlowMatchLCMScheduler`.
- **Why not `dpmpp_sde`** (recommended by some Civitai model cards): the Z-Image pipeline
  imposes custom `sigmas`, and `DPMSolverSDEScheduler.set_timesteps` does not accept them
  (it also needs the `torchsde` package). Same reason DPM++ 2M / DPM2a are not exposed.
  Note that ComfyUI's **`simple` scheduler == our default `sgm_uniform`**, and ComfyUI's
  **CFG 1.0 == our guidance 0** — so those model cards are already satisfied by the
  defaults.
- Files: `cz_pipeline.py` (`SAMPLER_CHOICES`, `_build_scheduler`), `README.md`,
  `config_modification_tutorial.txt`, `tests/test_xyz.py` (sampler suggestions now derive
  from `SAMPLER_CHOICES` instead of a hardcoded list).

## 1.12.0 — 2026-07-20 — X/Y/Z grid: compare LoRA files (epochs, versions)

Comparing several trainings of the same LoRA — epochs of one run, or successive CivitAI
versions — meant editing the Models panel and rebuilding a grid by hand. Two axes now do
it in one build.

- **`LoRA` axis**: swaps the *file* in LoRA slot 1 and keeps the weight set in the Models
  panel. Other active slots are left untouched. `None` is a valid value → control cell
  with no LoRA.
- **`LoRA + weight` axis**: varies both at once, written `name:weight`
  (`ollie_e10:0.6, ollie_e20:0.9`), for when the best weight differs per epoch. Split on
  the *last* `:` so Windows paths survive.
- **`⤵ suggest` lists the available LoRAs** (same mechanism as Checkpoint): the button
  drops the full list into the field, ready to prune. For `LoRA + weight` each entry is
  pre-filled with the current weight, so only the numbers need editing. The inserted list
  is CSV-quoted, so a filename containing a comma round-trips.
- Names resolve like every other closed list (`_xyz_match`): any unambiguous fragment
  works (`e000020`), ambiguous or unknown ones are rejected at **Build** time rather than
  mid-series.
- Cell labels show the base name without extension, truncated **from the left** — LoRA
  being compared usually differ only by their `_e000020` suffix, so trimming the end
  would have made every column read the same.
- Available from the CLI too: `--xyz "LoRA=ollie_e10, ollie_e20, None"`.
- Files: `cz_ui.py` (`_XYZ_AXES` + `lora_name` / `lora_name_weight` in `_xyz_suggestions`
  / `_xyz_validate_axis` / `_xyz_apply`, new `_xyz_fmt_value` + `_xyz_current_lora_weight`),
  `cz_cli.py` (`_xyz_cli_apply`, labels), `tests/test_xyz.py` (+6 tests: resolution,
  ambiguity, weights, apply, left-truncation, suggest round-trip).

## 1.11.4 — 2026-07-20 — Fix: asset-browser thumbnails corrupted files being served

Generating thumbnails while the Asset Browser SPA was displaying them produced
`h11 LocalProtocolError: Too much data for declared Content-Length` bursts in the console,
and broken images in the page. `FileResponse` takes `Content-Length` from an `os.stat`,
then re-reads the file to send it; `im.save(dst)` truncates `dst` to 0 and grows it, so a
request landing in that window declared one size and sent another. With 8 worker threads
over hundreds of files, the window was wide open.

- Thumbnails are now written to a temp file then `os.replace()`d (atomic): a reader sees
  either the previous complete file or the new one, never one mid-write. Same treatment
  for the other served files rewritten in place — `index.html`, `manifest.json` (both the
  reindex and the stub) and `<kind>.json`; `ab_open_fast` had the same race by design,
  since it spawns a background reindex that rewrites the manifest the SPA is polling.
- Side effect fixed: a truncated thumbnail kept a fresh mtime, so the
  `getmtime(thumb) >= getmtime(src)` check considered it up to date and it stayed corrupt.
- `os.replace` retries on Windows `PermissionError` (a destination held open by the
  serving thread), ~1 s with capped backoff; on definitive failure the thumbnail is
  counted as failed and regenerated next pass rather than written unsafely.
- Measured on a 1 writer / 4 reader race: 1725 Content-Length mismatches before, 0 after.
- Files: `cz_assetbrowser.py` (`_write_atomic_text`, `_replace_retry`, `_ab_make_thumb`).

## 1.11.3 — 2026-07-16 — Fix: LoRA hot-swap left stale adapters ("Already found a peft_config")

Switching LoRA in the UI logged a PEFT warning — *"Already found a `peft_config` attribute
in the model. This will lead to having multiple adapters."* — because
`unload_lora_weights()` does not reliably clear the transformer's `peft_config` in this
diffusers version. Since the hot-swap reuses the same adapter names (`cz_lora_i`), a stale
adapter could remain and the wrong LoRA be applied.

- `_apply_loras` now clears via a new `_clear_loras(pipe)`: `unload_lora_weights()` **then**
  an explicit `delete_adapters(get_list_adapters())` to remove any leftover adapter by name
  — so a swap A→B leaves only B registered, no accumulation.
- Safe by construction: the extra calls are wrapped in try/except and fall back to the
  previous behavior on any error.
- Files: `cz_pipeline.py` (`_clear_loras`), `tests/test_lora_hotswap.py` (+2 tests
  modelling the adapter lifecycle: a swap leaves only the new adapter; removing all clears
  the registry).

## 1.11.2 — 2026-07-16 — Fix: "Image number (batch)" was ignored in img2img / Input image

With **Input image** checked, `_ui_generate` called `run()` exactly once and returned a
single image — the **Image number (batch)** slider was silently dropped, so it only ever
worked in txt2img.

- The img2img/upscale branch now loops like txt2img: **n images**, seed **+1 per image**
  (or fixed if *Fix seed (no +1 per image)* is checked), **wildcards and random style
  re-rolled per image**, **Stop** honoured between images. The report lists each image
  (`1/4 (seed 1234)`); every image keeps its real saved filename for download.
- A **seed `-1`** is now resolved to a concrete value up front (as in txt2img), so
  **♻️ Reuse last seed** and the image metadata finally work in img2img too.
- **Refine (img2img) unchecked** = denoise 0 = no diffusion pass, so the output is
  deterministic and a batch would just write n identical files: the batch is clamped
  to 1 in that case (logged).
- Files: `cz_ui.py` (`_ui_generate`), `tools/smoke_test.py` (3 checks), `VALIDATION.md`.

## 1.11.1 — 2026-07-15 — Fix: SVDQuant/Nunchaku checkpoints were not filtered out

The README says FP8 / SVDQ (ComfyUI) checkpoints do not load in diffusers, and
`_safetensors_unsupported` filtered FP8 and `weight_scale`-style INT8/INT4 — but it
missed **SVDQuant / Nunchaku**, which uses a different convention: no `weight_scale`,
weights named `*.qweight`. Such a file stayed in the checkpoint dropdown and only failed
at load time.

- Detection added: any `*.qweight` tensor -> `"SVDQuant/Nunchaku INT4"`, skipped at
  startup with the reason like FP8. A normal BF16/FP16 checkpoint never has `qweight`,
  so there is no false positive.
- Verified on a real file (`…_svdqInt4R32Flux1Dev.safetensors`: 380 `qweight` keys,
  dtypes I32/BF16/I8, zero `weight_scale` — which is exactly why the old rule missed it)
  and against 9 other real checkpoints (BF16 -> kept, FP8 -> still caught).
- Files: `cz_pipeline.py` (`_safetensors_unsupported`).

## 1.11.0 — 2026-07-15 — "Rebuild ALL thumbnails (force)" button + parallel thumbnail generation

- New **🖼 Rebuild ALL thumbnails (force)** button in the Asset Browser header. It applies
  to the **tab you are on** — **Models**, **LoRAs** or **Outputs** — and force-regenerates
  every thumbnail from scratch (useful after a corrupt/partial thumbnail or a
  `thumbnail_size` change, which the normal "skip if up to date" rule would never redo).
- Runs **in the background with live progress**, reusing the same job + polling
  infrastructure as the CivitAI batch: a toast shows **`Thumbnails 42/177 — name`**, then
  a summary (`X rebuilt · Y failed · Z total`) and the tab reloads.
- **Thumbnail generation is now parallel** (`ThreadPoolExecutor`, `min(8, cpu)` by
  default, tunable via `asset_browser.thumb_workers`). PIL releases the GIL while
  decoding/resizing, so this speeds up the normal background indexing too, not just the
  new button.
- **Cache-busting**: rebuilt thumbnails keep the same URL, so the browser would have kept
  showing the old images — the SPA now appends a token after a rebuild.
- Defensive: a corrupt source counts as `failed` and the batch continues; a missing
  folder or a model with no preview yields no job instead of an error.
- The pre-existing Advanced ▸ Asset Browser "reindex" button (outputs only, synchronous)
  is unchanged.
- Files: `cz_assetbrowser.py` (`_ab_gen_thumbs` gains `force`/`progress`/`workers`,
  new `_thumb_jobs_for` + `rebuild_thumbs`, `_thumb_workers`), `cz_ui.py`
  (`_api_thumbs_rebuild` + `thumbs_rebuild` endpoint; the job registry and its endpoint
  are renamed `_BG_JOBS` / `job_progress` since they now serve three job types),
  `cz_assets.py` (button, handler, cache-buster), `tests/test_thumbs.py`.

## 1.10.2 — 2026-07-15 — Fix: the model SHA256 is cached (re-runs no longer re-hash the library)

`_compute_sha256` computed the hash but **never stored it**, so every batch pass re-read
every model in full just to obtain the same hash. Measured on a real library: **310 of
324 models have no `.metadata.json` sidecar → 416 GB re-read on each run.**

- The hash is now **persisted in `<name>.civitai.json`** (`sha256` + `sha256_size`) and
  reused. Lookup order: external `<name>.metadata.json` (Civitai-Helper convention) →
  our cache → compute (then cache).
- Cached **even when the model is unknown to CivitAI**, so those files stop being
  re-hashed on every pass too.
- **Invalidation**: the cache is rejected if the file size changed (model replaced /
  different version) → recompute.
- The fetch now **merges** the sidecar instead of overwriting it, so writing the CivitAI
  data no longer wipes the hash cache it had just saved. Sidecar writes (fetch +
  update-flag refresh) are now **atomic** (tmp + `os.replace`).
- `_needs_enrich` now tests `modelId` rather than "sidecar exists", so a sidecar holding
  only the cached hash is not mistaken for an enriched model.
- Net effect: the first pass still hashes what it must; **subsequent passes read no model
  bytes at all**.
- Files: `cz_civitai.py` (`_cached_sha256`, `_cache_sha256`, `model_sha256`, merged +
  atomic sidecar writes), `cz_civitai_batch.py` (`_needs_enrich`), `tests/test_civitai.py`
  (+4 tests: cache reused, stale-on-size-change, external sidecar wins, fetch keeps the
  cache).

## 1.10.1 — 2026-07-15 — Fix: example prompts were never fetched + API key ignored on some calls

Every CivitAI example was stored with an empty prompt (measured: **1130 / 1130**), so the
viewer showed "no prompt" for all of them.

- **Root cause**: examples came from the `/images` endpoint, which now returns
  **`"meta": null`** — CivitAI no longer publishes generation parameters there. The
  prompt was never in the response we were reading.
- **Fix**: the **`/model-versions/by-hash` response — which we already request — carries
  an `images` array with a *populated* `meta`** (prompt, steps, cfg, sampler…).
  `get_version_by_hash` now returns it and the fetch uses it, so prompts arrive with
  **zero extra requests** (`/images` is kept only as a fallback when a version has no
  showcase image). Verified end-to-end on a real model: **2/2 examples with prompt**.
- **API key was ignored on some calls**: `_api_get` only used a key when one was passed
  explicitly, so `get_latest_version` / `refresh_update_flag` (called by the batch with
  `api_key=None`) went out **anonymous** and missed gated/NSFW content. `_api_get` now
  falls back to the global key (UI → `preferences.json` → config).
- **HTTP errors are visible**: 401/403 (missing/invalid key) and 429 (rate limit) are now
  logged instead of being buried in debug — with a hint when no key is set.
- Missing prompts are now **honest**: examples carry a `has_prompt` flag and the viewer
  says *"the uploader did not publish the generation parameters for this image"* instead
  of implying a bug. The fetch message reports coverage (`3 example(s) (2 with prompt)`).
- **Backfilling existing sidecars**: previously fetched models have empty prompts. Re-run
  with `--all` to re-query metadata **without** re-downloading previews:
  `civitai_index.bat --kind all --all` (or `./civitai_index.sh --kind all --all`).
- Files: `cz_civitai.py` (`_api_get` key fallback + HTTP logging, `get_version_by_hash`
  images, new `_examples_from`), `cz_assetbrowser.py` / `cz_assets.py` (`has_prompt`),
  `tests/test_civitai.py` (+4 tests).

## 1.10.0 — 2026-07-15 — Negative LoRA weights + configurable weight range

The LoRA **Weight** sliders were hard-capped at `0..2`, so **negative weights were
impossible** — even though they are meaningful: a LoRA at a negative weight pushes *away*
from what it was trained on (a "skinny slider" at `-1` gives the opposite effect, an "age
slider" at `-0.5` swings the other way).

- Slider range is now **`-2..2` by default** and **configurable**:
  `"lora_weight_min": -2.0` / `"lora_weight_max": 2.0` in `config.txt`.
  Set `lora_weight_min` to `0` to forbid negatives.
- `default_lora_weight` is **clamped into the range**, so the slider can never start
  outside its own bounds.
- Defensive: non-numeric values, or `min >= max`, fall back to `-2..2` **and log why**
  (no silent surprise).
- The model layer never clamped weights (`set_loras`, X/Y/Z `LoRA weight` axis and the
  CLI `--lora NAME:WEIGHT` all pass floats straight through), so negatives work
  end-to-end — the UI slider was the only thing in the way.
- The LoRA panel now states the active range and that negatives invert the effect; both
  keys are documented in `config-sample.txt` and `config_modification_tutorial.txt`.
- Files: `cz_pipeline.py` (`_lora_weight_range`, `LORA_WEIGHT_MIN/MAX`, clamped
  `LORA_WEIGHT`), `cz_ui.py` (slider bounds + hint), `config-sample.txt`,
  `config_modification_tutorial.txt`, `tests/test_lora_weight_range.py`.

## 1.9.0 — 2026-07-15 — Switching Z-Image checkpoint reloads only the transformer

Same idea as the LoRA hot-swap (1.8.1), applied to the model itself. Switching from one
Z-Image checkpoint to another (**Z-Image checkpoint** dropdown, or the transformer
override) used to `free_vram()` and reload the **whole** pipeline — including the
**Qwen3-4B text encoder** and the VAE, which had not changed.

- When the **base repo and offload mode are unchanged** and only the **transformer**
  differs, `_ensure_base` now calls the new **`_swap_transformer`**: it loads *only* the
  new transformer and swaps it into the cached pipeline (`register_modules`), keeping the
  **VAE + Qwen3 text encoder + tokenizer + scheduler in VRAM**. The old transformer is
  freed (`empty_cache`).
- Covers all the "Z-Image → Z-Image" moves: single-file ↔ single-file, single-file →
  base repo's own transformer (clearing the override), and repo-subfolder overrides.
- Consistency taken care of: derived img2img/inpaint pipes (`from_pipe`) pointed at the
  **old** transformer → `_DERIVED` is cleared (rebuilding is free, weights are shared);
  LoRA adapters lived on the old transformer → they are **re-applied** to the new one.
  Under CPU offload the accelerate hooks are removed and re-attached around the swap.
- Safe fallback: any failure logs and falls back to the previous full reload.
  **Changing the base repo still reloads everything** (VAE/encoder genuinely change).
- New shared `_load_transformer()` used by both the full load and the swap.
- Files: `cz_pipeline.py` (`_load_transformer`, `_swap_transformer`, `_ensure_base`,
  `set_zimage_transformer`, `set_zimage_model`), `cz_ui.py` (status wording),
  `tests/test_model_swap.py` (7 tests incl. regression guards: a single-file switch must
  not free the pipe; a base-repo change must still free it).

## 1.8.1 — 2026-07-15 — Fix: switching a LoRA no longer reloads the whole model

Enabling / changing / removing a LoRA used to **reload the entire Z-Image pipeline**
(transformer + VAE + **Qwen3-4B text encoder**) — tens of seconds for what should be
instant, even though the model was already in VRAM.

- Cause: `set_loras()` called `free_vram()` (wiping `_BASE_PIPE`), and the base cache key
  included `tuple(LORAS)`, so any LoRA change invalidated the loaded pipeline.
- Fix: LoRAs are now **hot-swapped on the cached pipe** via the PEFT backend
  (new `_apply_loras`), and the cache key is back to `(repo, transformer, offload)`:
  - **weight-only change → `set_adapters`**, instant, nothing re-read from disk;
  - **different LoRA set → `unload_lora_weights` + reload of the LoRA files only** (~1 s);
  - derived pipes (img2img / inpaint, built with `from_pipe`) share the transformer, so
    they follow automatically.
- Safe fallback: if the hot-swap raises (e.g. missing PEFT backend), `_ensure_base` falls
  back to the previous full-reload path, so behaviour is never worse than before.
- Model/transformer/offload changes still reload, as they must.
- Files: `cz_pipeline.py` (`_apply_loras`, `_APPLIED_LORAS`, `set_loras`, `_ensure_base`,
  `free_vram`), `tests/test_lora_hotswap.py` (8 tests incl. regression guards: `set_loras`
  must not free the pipe, the cache key must not contain the LoRAs).

## 1.8.0 — 2026-07-14 — Batch CivitAI enrichment (.bat/.sh script + "Fetch all" button + new-version warnings)

Enrich a whole folder at once instead of one model at a time, from the UI **or** from a
standalone script you can run in parallel.

- **Standalone `cz_civitai_batch.py`** (imports no torch → starts instantly). Scans the
  LoRA / checkpoint folders and fetches the **missing** CivitAI info for each model
  (preview + trigger words + **example prompts**), skipping ones already done but still
  **refreshing the "newer version" flag**.
  ```
  python cz_civitai_batch.py --kind {loras,models,all} [--force] [--all]
         [--shard i/m] [--sleep 0.5] [--api-key KEY]
  ```
  `--shard i/m` splits the file list into disjoint subsets so **several processes can run
  in parallel**. Prints a per-model progress line + a final `enriched/skipped/updated/
  failed` summary; non-zero exit only if everything failed.
- **Wrappers**: `civitai_index.bat` / `.sh` (pass-through args, finds the venv Python,
  forces UTF-8) and `civitai_index_parallel.bat` / `.sh` (`[N]`, default 4) that launch
  **N parallel shards** — this is the intended "batch in parallel" workflow.
- **"🔄 Fetch all missing" button** in the Asset Browser (LoRAs / Models tabs): runs the
  same core in a background thread with a live toast (`Batch 12/48 — name…`), then a
  summary and catalog reload. New `civitai_fetch_all` Gradio endpoint (polled via the
  existing `civitai_progress`).
- **New-version warnings**: `fetch` and the batch now compare the local version to the
  latest on CivitAI (`get_latest_version`) and store `update_available` +
  `latest_versionName` in `<name>.civitai.json`. The Asset Browser shows a **⚠ update**
  badge on the card and a "Newer version on CivitAI: …" line in the lightbox.
- **Example prompts**: already captured since 1.7.2; the batch path reuses the same fetch,
  so they are filled in bulk too.
- **Config** `"civitai_batch": {"enabled": true, "sleep": 0.5, "check_updates": true}` —
  `enabled:false` hides the "Fetch all" button (the per-model 🔎 still works); `sleep` is
  rate-limit friendly; `check_updates:false` skips the extra version request.
- Files: `cz_civitai_batch.py` (new), `cz_civitai.py` (`get_latest_version`,
  `refresh_update_flag`, `update_available` in the sidecar), `cz_ui.py`
  (`civitai_fetch_all` endpoint), `cz_assetbrowser.py` (catalog `update`/`latest`, SPA
  render flag), `cz_assets.py` (button + toast + badge + version line),
  `civitai_index.bat/.sh`, `civitai_index_parallel.bat/.sh`, `config-sample.txt`,
  `tests/test_civitai_batch.py`.
- **Rate limits**: CivitAI throttles; keep parallel shards modest and set a CivitAI API key
  (Advanced) for heavy runs.

## 1.7.3 — 2026-07-14 — Jump to a LoRA / checkpoint in the Asset Browser (🖼️ icon)

Fooocus2026-style shortcut: a small **🖼️ icon** sits next to each **LoRA** dropdown
(**Advanced ▸ LoRA**) and next to the **Z-Image checkpoint** dropdown (**Advanced ▸
Models**). Clicking it opens the **Asset Browser in a new tab, already on the right source
tab and focused on that item** — its lightbox (preview + trigger words + example images
from 1.7.1/1.7.2) opens immediately.

- The browser is opened at `index.html?src=loras|models&focus=<file>`; the SPA reads the
  query on load, switches source, clears the folder filter and opens the matching card.
- The catalog is (re)built synchronously before the tab opens so the target is present.
- Base HF repos (Turbo/Base — no local file) just open the **Models** tab (nothing to
  focus). `None` LoRA slots open the **LoRAs** tab.
- Files: `cz_ui.py` (`_asset_focus_url` + 🖼️ buttons wired to each LoRA / the checkpoint),
  `cz_assets.py` (query parsing + `_tryFocus`).

## 1.7.2 — 2026-07-14 — CivitAI fetch: live progress + example viewer with prompts

Two UX fixes on the Asset Browser's **🔎 Fetch from CivitAI** button (1.7.1).

- **Live progress instead of a silent freeze.** The fetch now runs in a background
  thread and the model lightbox shows a status line with a spinner + progress bar that
  advances through the real phases: **`Hashing model file… 42%`** (a *real* byte-percentage
  — the only slow step, and only when there is no `<name>.metadata.json` sidecar) →
  `Querying CivitAI…` → `Fetching example images…` → `Downloading preview…`, then an inline
  ✅/⚠️ result (no more blocking `alert()`). The button is disabled while it runs.
  New Gradio endpoint `civitai_progress`; the client polls it every ~400 ms.
- **Example images are now clickable.** Each CivitAI example opens a full-screen viewer
  showing the image **large** with its **generation prompt** underneath (+ **Copy prompt**
  and *Open image*), and **← / →** (mouse or keyboard) to browse between examples. The
  example prompts were already downloaded into `<name>.civitai.json` — the catalog now
  carries them through (`{url, prompt, width, height}`) instead of the URL alone.
- No new dependency, no new config; purely additive to the existing button. Robust:
  any error is shown inline and never blocks the browser.
- Files: `cz_civitai.py` (`fetch_civitai_for_model(progress=…)` + real hash %),
  `cz_ui.py` (threaded job registry + `civitai_progress` endpoint),
  `cz_assetbrowser.py` (keep example prompts in the catalog),
  `cz_assets.py` (status bar, polling, example viewer + CSS).

## 1.7.1 — 2026-07-12 — Asset Browser: CivitAI enrichment (previews / trigger words / examples)

- New **`cz_civitai.py`** (technique from Fooocus2026): looks a model/LoRA up on **CivitAI
  by its SHA256** — read from the sibling `<name>.metadata.json` when present, so multi-GB
  checkpoints are **not** re-hashed — then fetches **trigger words** + top **example images**
  and saves `<name>.preview.png` (the sidecar convention the Asset Browser already scans) +
  `<name>.civitai.json`.
- **Asset Browser** (LoRAs / Models tabs): a **🔎 Fetch from CivitAI** button in the model
  lightbox (a `civitai_fetch` Gradio API endpoint) downloads the preview + trigger words,
  rebuilds the catalog and reloads — the placeholder becomes a real preview. The lightbox
  now shows **example images** + a **CivitAI page** link; the catalog reads trigger words
  from `<name>.civitai.json` (falling back to the safetensors header).
- Optional **CivitAI API key** — paste it in **Advanced > CivitAI access** (saved to
  `preferences.json`) or set `civitai_api_key` in config; for gated/NSFW previews and to
  avoid rate limits. Most public models work without one.
- Files: `cz_civitai.py`, `cz_assetbrowser.py`, `cz_ui.py`, `cz_assets.py`, `cz_core.py`
  (`APP_VERSION` 1.7.1), `config-sample.txt`, `config_modification_tutorial.txt`.

## 1.7.0 — 2026-07-12 — Presets, seed reuse, Advanced tab, PNG Info, a1111 metadata, Asset Browser overhaul

A large UI/UX pass (all in the `cz_*` modules).

- **Presets (Fooocus-style)** — new *⭐ Presets* accordion (Settings). A preset bundles
  prompt/negative, styles, size, steps/CFG, sampler/schedule, image number, checkpoint,
  transformer override and LoRAs into `presets/<name>.json`. **Load** applies the widgets
  AND the model/LoRAs (a chained silent checkpoint apply keeps the preset's steps/CFG);
  **Save as new / Update selected / Delete / refresh**. `presets/` gitignored except
  `example.json`.
- **Seed management** — *♻️ Reuse last seed* button (refills the field with the previous
  render's real seed) + *Fix seed (no +1 per image)* toggle. A `-1` random seed is now
  resolved to a concrete value before generation, so the metadata stores the real seed
  (previously it saved `-1`).
- **Advanced tab** — new *Advanced* tab (after Save) for advanced settings; the *Hugging
  Face access (gated models)* block moved here from Models.
- **Input Image → PNG Info** — a "Read prompt / metadata from an image" reader (a filepath
  uploader that preserves PNG chunks) parses crispz, **A1111/Civitai** (`parameters`) and
  ComfyUI metadata, with *Send prompt* / *Send seed* to the fields.
- **Metadata scheme** (`metadata_scheme`, Advanced > Metadata) — `crispz` (default) or
  `a1111`, which also writes an A1111/Civitai `parameters` PNG chunk so **Civitai reads the
  prompt/seed/params** on upload (crispz chunk + sidecar kept in both).
- **Read wildcards in order** (`wildcards_in_order`, Advanced > Generation) — a batch sweeps
  each wildcard file line by line (deterministic) instead of picking random lines.
- **Also save pre-upscale image** (`save_pre_upscale`) — in txt2img + auto-upscale, also
  save the base txt2img image (before ESRGAN/refine), tagged `txt2img`.
- **Configurable LoRA slots** (`lora_slots`, default 3) — 1–10 slots; a live slider in
  Advanced > Generation shows/hides them (persisted in `preferences.json`).
- **Asset Browser overhaul** — the output gallery now opens as a **standalone page in a new
  tab** via a button; **instant open** (manifest written immediately, thumbnails generated
  in the background behind a shimmer placeholder that swaps to the real thumbnail); images
  save into **`out/YYYY-MM-DD/`** date subfolders (`date_subfolders`, recursive scan);
  **per-image delete**; a **subfolder sidebar** with counts, per-folder **hide** and a
  **Hidden** toggle (persisted in localStorage), defaulting to the current day; **keyword
  search** over the embedded metadata; and **Outputs / LoRAs / Models** source tabs
  (LoRAs/Models show a Civitai preview if one sits next to the `.safetensors`, else a
  placeholder + trigger words).
- Files: `cz_ui.py`, `cz_assets.py`, `cz_assetbrowser.py`, `cz_imageio.py`, `cz_prompt.py`,
  `cz_pipeline.py`, `cz_cli.py`, `cz_core.py` (`APP_VERSION` 1.7.0), `config-sample.txt`,
  `config_modification_tutorial.txt`, `presets/example.json`.

## 1.6.0 — 2026-07-07 — Model-loading progress in the terminal and UI

The first model load downloads from Hugging Face and then reads several GB into VRAM —
previously a long silent gap (the report was `317.3s` with no sign of progress). The
blocking `from_pretrained` now runs in a daemon thread while a heartbeat (every ~2 s)
reports where the load is:

- **Terminal**: a single rewritten line `[crispz][load] Z-Image base... 45s | 3.2 GB in
  VRAM` (during the first-run download, before anything is allocated, it reads
  `... 12s (downloading / reading, first run only)`).
- **UI**: the Gradio progress bar advances — honest %, based on **VRAM allocated /
  `target_vram_gb`** once loading into memory starts (capped 95 %), a small time-based
  bar during the download phase.
- Applied to the three heavy loads: **Z-Image base**, the **single-file transformer**
  (Civitai checkpoint), and **Z-Image Omni**.
- **Zero-cost off**: `"load_progress": {"enabled": false}` loads directly with no monitor
  thread. `target_vram_gb` (default 14) and `heartbeat_s` (default 2) are tunable.
- The monitor never swallows errors — a failed load re-raises exactly as before.
- Files: `cz_pipeline.py` (`_load_monitor` + pure `_fmt_load`/`_load_pct`), `cz_core.py`
  (`APP_VERSION` 1.6.0), `config-sample.txt`, `tests/test_load.py`.

## 1.5.2 — 2026-07-05 — Fix: empty "Apply override" no longer clears the checkpoint

- Selecting a checkpoint in **Z-Image checkpoint** applies it automatically. Clicking the
  transformer-override **Apply** button with an **empty** field used to call
  `set_zimage_transformer("")`, silently wiping that selection (the terminal then showed
  `transformer -> (repo de base)` and Generate loaded the plain base repo).
- The button is now a no-op on an empty field (returns a clear hint instead of clearing),
  and was relabeled **"Apply override"** (secondary) to distinguish it from the main
  checkpoint dropdown. To go back to the plain base repo, pick an official repo in the
  dropdown.

## 1.5.1 — 2026-07-05 — Fix: tensor-size mismatch on non-/32 image dimensions

- Fixes `Upscale/img2img failed: The size of tensor a (150) must match the size of
  tensor b (148)` — hit e.g. with **Force aspect ratio** crops whose height/width is a
  multiple of 16 but not 32 (1200, 848…). The Z-Image transformer patchifies the VAE
  latent by 2, so **every pixel dimension must be a multiple of 32**.
- `round_to_multiple` default is now **32** (txt2img sizes, refine tiles, ESRGAN targets
  all align), and `_refine_whole` snaps its input to /32 (resize) before diffusion then
  restores the original size — callers and tiled overlap-add contracts unchanged.

## 1.5.0 — 2026-07-05 — X/Y/Z grid in the CLI

The comparison grid is no longer UI-only: `--xyz "AXIS=v1,v2,…"` (repeat up to 3 times
for X, Y, Z) with `--txt2img` runs every combo and ends with the same annotated contact
sheet(s) in `<output>/xyz_<timestamp>/` (paths printed on stdout).

```bash
python app.py --cli --txt2img --prompt "a red cat" \
    --xyz "Steps=4,8,12" --xyz "Guidance=0, 3.5" --save-mode local
```

- Same axes and validation as the UI grid (shared helpers): case-insensitive axis and
  closed-list resolution (`step` → `Steps`, `uni` → `unipc`), quotes protect commas,
  Prompt S/R checked against `--prompt`, duplicate axes rejected, `max_jobs` cap.
  Upscale-only axes (ESRGAN model, Factor, Denoise, Tile, Refine tile) require
  `--upscale` (clear error otherwise).
- Each combo is saved as a normal output (tag `xyz`, metadata includes the combo);
  **Ctrl+C assembles a partial sheet** with the cells rendered so far.
- Respects `xyz_grid.enabled` (config) — disabled = clear error, nothing runs.
- Ready-to-run example scripts: `xyz_example.bat` / `xyz_example.sh`
  (`xyz_example.bat "your prompt"` → 2×2 Steps × Guidance grid; edit the `--xyz` lines
  to change the axes). Fails loudly with a non-zero exit code on error.
- Files: `cz_cli.py` (`--xyz`, runner), `cz_ui.py` (axes table gains abstract `param`
  names shared with the CLI), `xyz_example.bat`/`.sh`, `tests/test_xyz.py`
  (CLI apply + axis resolution).

## 1.4.0 — 2026-07-05 — Tag autocomplete in prompt fields

Type-ahead suggestions in the **prompt** and **negative prompt** fields.

- **Sources**: CSVs listed in `tag_autocomplete.sources` are downloaded **once at first
  launch** into `tags/` (atomic tmp+rename, one-line console progress); any `.csv` you
  drop into `tags/` becomes a source too (rich `name,category,count,"aliases"` format or
  one word per line). Local assets are merged in: your **wildcards** appear as
  `__name__` entries at top priority.
- **Client**: vanilla JS injected only when enabled (`gr.Blocks(head=…)`). Index built
  once — global popularity sort, cross-source dedup, **2-char prefix buckets, early
  exit** — then a dropdown under the caret: ↑/↓ navigate, **Tab/Enter** insert (current
  comma-delimited token replaced, underscores → spaces, `__wildcards__` kept verbatim),
  **Escape** closes. Aliases match too (shown in gray with the matched alias). Startup
  and per-keystroke timings logged in the browser console (`[tagac] ready in N ms`,
  rolling average per 50 keystrokes).
- **Zero-cost off**: `"tag_autocomplete": {"enabled": false}` → `cz_tags` never imported,
  nothing downloaded, no script injected.
- New generic helper `cz_core.download_with_progress` (atomic, 64 KB blocks, one-line
  progress) — also used by the inswapper/GFPGAN downloads from this version on.
- Files: `cz_tags.py` (new), `cz_assets.py` (`TAG_AC_JS`), `cz_ui.py`, `cz_core.py`,
  `cz_cli.py` (`tags/` served), `config-sample.txt`, `.gitignore` (`tags/`),
  `tests/test_tagac.py`.

## 1.3.0 — 2026-07-05 — Contextual suggestions for X/Y/Z value fields

- Each value field adapts to the axis picked in the neighboring dropdown: the
  **placeholder** shows contextual examples, and a **`⤵ suggest`** button inserts a
  ready-to-prune list — app lists for closed choices (Sampler, Schedule, Performance,
  Checkpoint incl. both folders, ESRGAN models), classic calibration values for numeric
  axes (Steps `4, 8, 12, 20, 28`, Guidance `0, 2, 3.5, 5`, Denoise `0.2, 0.3, 0.4`…),
  syntax hint for Prompt S/R.
- The fill button never overwrites a non-empty field; values containing commas/quotes are
  CSV-quoted so the inserted text re-parses exactly (round-trip tested).
- Case-insensitive partial matching at build time (from 1.2.0) completes the loop:
  suggestions can be shortened by hand (`uni` → `unipc`).
- Config: sub-key `"suggest": true` of the `xyz_grid` block; `false` = no buttons, no
  handlers, static placeholders.
- Files: `cz_ui.py`, `config-sample.txt`, `tests/test_xyz.py`.

## 1.2.0 — 2026-07-05 — X/Y/Z comparison grid

Compare parameter variations on an annotated contact sheet, powered by the job queue.

- **X/Y/Z grid panel** (accordion under the Job queue): pick 1–3 axes and their values
  (comma-separated; quotes protect commas). **Build grid → queue** turns every combo
  into a queued job; run/pause/reorder like any other jobs.
- **Axes**: Checkpoint, Sampler, Schedule, Steps, Guidance, Seed, ESRGAN model, Factor,
  Denoise, Tile, Refine tile, LoRA weight (applies to all active LoRAs), **Performance**
  (applies the whole preset), **Prompt S/R** (a1111-style search & replace: first value =
  search term, next values = replacements; validated against the prompt at build time).
- **Validation at build**: numeric casts, closed lists resolved case-insensitively (unique
  substring accepted, e.g. `uni` → `unipc`), duplicate axes rejected, combo count capped
  (`max_jobs`, default 100).
- **Contact sheets** (Pillow, no new dependency): one annotated sheet per Z value — X in
  columns, Y in rows, letterboxed cells (`thumb`, default 512 px), missing cells drawn as
  placeholders — saved under `<output>/xyz_<timestamp>/` and appended to the result
  gallery. Cells are accumulated across pause/resume, so a paused grid still ends with a
  complete sheet.
- Config block: `"xyz_grid": {"enabled": true, "max_jobs": 100, "thumb": 512}` (requires
  `job_queue`); `enabled=false` creates nothing (zero cost).
- Files: `cz_ui.py` (axes table, validation, plan builder, assembler, panel),
  `config-sample.txt`, `tests/test_xyz.py`.

## 1.1.0 — 2026-07-04 — Job queue

Queue up generations with different settings and run them unattended (e.g. overnight).

- **`+ Queue`** snapshots ALL current settings: the full Generate parameter set **plus the
  global model state** (checkpoint/transformer, active LoRAs + weights, sampler, schedule),
  so each job is self-contained and reproducible regardless of what is loaded later.
  The button label shows the pending count (`+ Queue (3)`).
- **Job queue panel** (accordion under the prompt area): readable labels
  (`txt2img · model · 1024x768 · 8 steps · seed 42 · x2 · "prompt…"`), select a job and
  **Up / Down / Remove / Clear**.
- **`Run queue`** executes jobs in order in the normal progress window; the session
  history and saved outputs accumulate as usual. Before each job the model state is
  restored through the existing setters, so **VRAM is purged automatically only when the
  model actually changes** between jobs (zero cost otherwise).
- **Stop pauses the queue**: the current job is interrupted (existing Stop behavior) and
  the remaining jobs stay queued — press `Run queue` again to resume. A failing job is
  logged (`[crispz][queue] …`) and the queue continues with the next one.
- Config block (`config.txt`): `"job_queue": {"enabled": true}` — set `false` to remove
  the panel entirely (no components, no handlers, zero cost).
- Files: `cz_ui.py` (panel + handlers + pure helpers), `cz_core.py` (`APP_VERSION`,
  module-prefixed logs), `config-sample.txt`, `tests/test_queue.py`.
- Limits (v1): the queue lives in memory (cleared on page reload); jobs are not editable
  in place (remove + re-queue); execution is sequential.

## 1.0.0 — 2026-07-04 — Baseline

Everything up to and including: unified Inpaint/Outpaint editor (brush / expand sides /
reframe, ~1 MP bound, harmonize), auto-upscale after generate, local BLIP captioner +
auto-describe, unified Z-Image checkpoint dropdown (+ extra folder, Performance
auto-sync), multi-LoRA, face swap + GFPGAN, remove background, Asset Browser (instant
open, day filter, placeholders), Ollama integration with offline fallbacks, CLI and
server mode.
