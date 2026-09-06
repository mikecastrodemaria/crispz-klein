#!/usr/bin/env python3
"""Check whether the FLUX.2 Klein extras crispz-klein is still missing have shipped.

Unlike the upstream forks, klein needs NO edit model to watch for: multi-reference
editing is native to the base pipeline. What it actually lacks (cf. FORK.md) is:

  - a ControlNet for FLUX.2 Klein (pose / depth / lineart);
  - an edit-task LoRA (the Qwen-Image-Edit catalogue cannot load on FLUX.2, so
    cz_edit_loras.EDIT_LORA_SPECS ships empty).

Exit code 0 if at least one candidate exists on Hugging Face, 1 otherwise.
Prints a one-line status. Used by the daily watcher (and runnable by hand).
No deps beyond stdlib.
"""
import json
import sys
import urllib.parse
import urllib.request

# Recherches HF, pas des repos figes: personne ne sait sous quel nom ces modeles
# sortiront. On interroge l'API de recherche et on filtre sur le nom.
QUERIES = [
    ("ControlNet FLUX.2 Klein", "flux.2 klein controlnet"),
    ("Edit LoRA FLUX.2 Klein", "flux.2 klein edit lora"),
]


def search(q, limit=5, timeout=10):
    url = ("https://huggingface.co/api/models?search="
           + urllib.parse.quote(q) + f"&limit={limit}&sort=downloads&direction=-1")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "crispz-watcher"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return [m.get("id") for m in json.load(r)]
    except Exception:
        return []          # reseau / 4xx -> traite comme "rien trouve"


def main():
    found = []
    for label, q in QUERIES:
        hits = search(q)
        if hits:
            found.append(f"{label}: " + ", ".join(hits[:3]))
    if found:
        print("CANDIDATES: " + " | ".join(found)
              + " -> check them, then declare in cz_edit_loras.EDIT_LORA_SPECS")
        return 0
    print("not yet: no FLUX.2 Klein ControlNet or edit LoRA found on the Hub")
    return 1


if __name__ == "__main__":
    sys.exit(main())
