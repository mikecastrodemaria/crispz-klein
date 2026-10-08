"""Point C of FORK.md: klein-4B is distilled (~4 steps). The diffusers doc says
"For step-wise distilled models, guidance_scale is ignored". Should that be true, the
UI's "guidance" slider is inoperative and debate A (negative_prompt_embeds)
is settled outright.

Renders the SAME seed at guidance_scale 1.0 / 4.0 / 8.0 and compares the pixels.
    identical -> the guidance is ignored -> supports.negative = False, an empty _cfg
    different -> the guidance is active  -> code the negative_prompt_embeds

"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from diffusers import Flux2KleinPipeline

REPO = "black-forest-labs/FLUX.2-klein-4B"
PROMPT = ("a comic book panel, a woman with red hair running in the rain at night, "
          "ink lines, flat colors, dramatic angle")
SEED, STEPS, W, H = 7, 4, 1024, 1024
SCALES = [1.0, 4.0, 8.0]


def main():
    # A GPU bench, not a unit test: it downloads the real 4B checkpoint and renders
    # three 1024x1024 images. Without CUDA there is nothing to measure, and running
    # it on a CI runner would just download ~15 GB to fail on .to("cuda").
    if not torch.cuda.is_available():
        print("SKIP test_klein_guidance (needs CUDA and the real 4B checkpoint)")
        return 0
    print(f"loading {REPO} ...", flush=True)
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    print(f"loaded in {time.time()-t0:.1f}s  "
          f"vram={torch.cuda.memory_allocated()/1024**3:.1f} GB", flush=True)

    out = {}
    for g in SCALES:
        gen = torch.Generator("cuda").manual_seed(SEED)
        t = time.time()
        img = pipe(prompt=PROMPT, height=H, width=W, num_inference_steps=STEPS,
                   guidance_scale=g, generator=gen).images[0]
        dt = time.time() - t
        path = f"tests/out_guidance_{g}.png"
        img.save(path)
        out[g] = np.asarray(img).astype(np.int16)
        print(f"guidance={g:<5} {dt:6.2f}s  -> {path}", flush=True)

    print("\n--- ecarts pixel vs guidance=1.0 ---")
    ref = out[SCALES[0]]
    verdict_ignored = True
    for g in SCALES[1:]:
        d = np.abs(out[g] - ref)
        mae, mx = d.mean(), d.max()
        print(f"guidance={g:<5} MAE={mae:8.4f}  max={mx:4d}")
        if mae > 1.0:
            verdict_ignored = False

    print()
    if verdict_ignored:
        print("VERDICT: guidance_scale IGNORE (modele distille).")
        print("  -> _cfg() passes neither guidance_scale nor negative")
        print("  -> caps supports.negative = False + a warning on a spec with a negative")
    else:
        print("VERDICT: guidance_scale ACTIF.")
        print("  -> encode negative_prompt_embeds through pipe.encode_prompt() in _cfg")
    return 0 if True else 1


if __name__ == "__main__":
    sys.exit(main())
