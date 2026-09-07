"""Portage FLUX.2 Klein : les 4 chemins du protocole v1 sur le vrai GPU.

    gen      -> generate()        Flux2KleinPipeline
    edit     -> generate_omni()   Flux2KleinPipeline, MEME objet (multi-ref natif)
    inpaint  -> inpaint_run()     Flux2KleinInpaintPipeline
    upscale  -> _refine_whole()   Flux2KleinInpaintPipeline + masque blanc injecte
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Le repo de base est PIN sur le 4B avant l'import de cz_pipeline. Sans ca, ce test
# suit le dernier modele choisi dans l'UI (persiste dans preferences.json depuis
# 1.19.0): choisir le 9B pour un projet faisait soudain telecharger 35 Go et
# reclamer ~29 Go de VRAM a la suite de tests, qui echouait pour une raison qui
# n'a rien a voir avec le code. Mettre KLEIN_E2E_MODEL pour viser une autre base.
os.environ["KLEIN_MODEL"] = (os.environ.get("KLEIN_E2E_MODEL")
                             or "black-forest-labs/FLUX.2-klein-4B")

import torch
from PIL import Image, ImageDraw
import cz_pipeline as p

P1 = ("a comic book panel, a woman with red hair running in the rain at night, "
      "ink lines, flat colors, dramatic angle")


def vram():
    return torch.cuda.memory_allocated() / 1024**3


def step(name, fn):
    t = time.time()
    out = fn()
    print(f"  {name:<22} {time.time()-t:6.2f}s   vram={vram():5.2f} GB", flush=True)
    return out


def main():
    ok = True
    print(f"base repo: {p.BASE_REPO}"
          f"{'  (KLEIN_E2E_MODEL)' if os.environ.get('KLEIN_E2E_MODEL') else ''}")

    print("\n[1] gen (txt2img)")
    img = step("generate", lambda: p.generate(P1, 1024, 1024, 4, 7))
    img.save("tests/e2e_1_gen.png")
    assert img.size == (1024, 1024), img.size

    print("\n[2] edit - 1 reference")
    e1 = step("generate_omni x1", lambda: p.generate_omni(
        [img], "make it daytime, sunny, blue sky", "", 1024, 1024, 4, 7))
    e1.save("tests/e2e_2_edit1.png")

    print("\n[3] edit - 2 references (multi-ref natif)")
    ref2 = Image.new("RGB", (1024, 1024), (250, 240, 200))
    ImageDraw.Draw(ref2).ellipse((300, 300, 700, 700), fill=(220, 40, 40))
    e2 = step("generate_omni x2", lambda: p.generate_omni(
        [img, ref2], "put the character from the first image in front of the second image",
        "", 1024, 1024, 4, 7))
    e2.save("tests/e2e_3_edit2.png")

    print("\n[4] inpaint (zone blanche du masque)")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle((620, 60, 980, 380), fill=255)
    ip = step("inpaint_run", lambda: p.inpaint_run(
        img, mask, "a large full moon behind heavy clouds", 4, 0.9, 7))
    ip.save("tests/e2e_4_inpaint.png")

    print("\n[5] img2img via masque blanc injecte (_refine_whole)")
    pipe = p.get_pipe("img2img")
    print(f"  pipeline img2img       {type(pipe).__name__}")
    assert type(pipe).__name__ == "Flux2KleinInpaintPipeline", type(pipe).__name__
    r = step("_refine_whole d=0.35", lambda: p._refine_whole(pipe, img, 0.35, 4, P1, 7))
    r.save("tests/e2e_5_refine.png")
    assert r.size == img.size, (r.size, img.size)

    print("\n[6] partage du pipeline derive (pas de VRAM en double)")
    same = p.get_pipe("inpaint") is p.get_pipe("img2img")
    print(f"  img2img is inpaint     {same}")
    ok &= same
    omni_is_base = p.get_pipe("omni") is p.get_pipe("txt2img")
    print(f"  omni    is base        {omni_is_base}")
    ok &= omni_is_base

    print(f"\nVRAM totale finale: {vram():.2f} GB (un seul modele charge)")
    print("RESULTAT:", "OK" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
