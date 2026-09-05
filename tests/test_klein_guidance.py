"""Point C de FORK.md : klein-4B est distille (~4 steps). La doc diffusers dit
"For step-wise distilled models, guidance_scale is ignored". Si c'est vrai, le
curseur "guidance" de l'UI est inoperant et le debat A (negative_prompt_embeds)
est tranche d'office.

Rend la MEME seed a guidance_scale 1.0 / 4.0 / 8.0 et compare les pixels.
    identique  -> guidance ignore  -> supports.negative = False, _cfg vide
    different  -> guidance actif   -> coder les negative_prompt_embeds
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
        print("  -> _cfg() ne passe ni guidance_scale ni negative")
        print("  -> caps supports.negative = False + warning sur un spec avec negative")
    else:
        print("VERDICT: guidance_scale ACTIF.")
        print("  -> coder negative_prompt_embeds via pipe.encode_prompt() dans _cfg")
    return 0 if True else 1


if __name__ == "__main__":
    sys.exit(main())
