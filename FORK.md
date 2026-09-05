# crispz-klein — fork FLUX.2 Klein de crispz-qwen-edit

Fork **texte → image + édition multi-référence** basé sur
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
(4B, Apache 2.0, distillé ~4 steps). Parti de **crispz-qwen-edit** plutôt que de
crispz-studio ou crispz-krea2, parce que klein-4B a le même profil fonctionnel que
Qwen-Image-Edit — txt2img, img2img, inpaint et édition multi-référence dans un seul
fork — là où krea2 est le fork amputé (`img2img: False`, `inpaint: False`, pas
d'omni, pas de single-file).

- Modèle : `black-forest-labs/FLUX.2-klein-4B` (défaut). **Repo public, Apache 2.0.**
- Remotes : `origin` = crispz-klein, `upstream` = crispz-studio, `qwen` = crispz-qwen-edit.
- Base du fork : `3c128c8` (crispz-qwen-edit/main).

> **État : portage non commencé.** Ce fichier est le plan de portage, pas la
> description d'un fork terminé. L'arbre est encore intégralement Qwen-Image.
> Les numéros de ligne renvoient à `cz_pipeline.py` tel qu'hérité de `3c128c8`.

## Pourquoi klein-4B plutôt que Krea 2 ou Anima

| | Qwen-Image-Edit-2511 | **klein-4B** | Krea 2 | Anima turbo |
|---|---|---|---|---|
| Params | 20B | **4B** | 12,9B | 2B |
| VRAM bf16 | ~40 Go | **~13 Go** | ~26 Go | ~5 Go |
| Steps | ~8 (Lightning) | **4** | 8 | 8–12 |
| `from_single_file` | oui | **oui** | non | n/a (ComfyUI) |
| img2img / inpaint | oui | oui (via inpaint) | **non** | non |
| multi-référence | modèle séparé | **natif, même pipeline** | non | LoRA expérimental |
| Licence | Apache 2.0 | **Apache 2.0** | Community (plafond 1 M$) | — |

## Ce que diffusers expose

```
diffusers/pipelines/flux2/
  Flux2Pipeline              (FLUX.2-dev, hors scope)
  Flux2KleinPipeline         base : txt2img ET édition multi-réf, unifié
  Flux2KleinInpaintPipeline  inpaint, et le SEUL à exposer `strength`
  Flux2KleinKVPipeline       variante KV-cache (repo -9b-kv, hors scope)
```

Signatures vérifiées sur `diffusers/main` :

| kwarg | `Flux2KleinPipeline` | `Flux2KleinInpaintPipeline` |
|---|---|---|
| `image` | **`list[PIL] \| PIL`** | oui |
| `image_reference` | — | oui |
| `mask_image` | — | oui |
| `strength` | **absent** | oui (défaut 0.8) |
| `padding_mask_crop` | — | oui |
| `guidance_scale` | 4.0 | 8.0 |
| `negative_prompt` | **absent** | **absent** |
| `negative_prompt_embeds` | oui | oui |
| `sigmas` | oui | oui |

`Flux2KleinPipeline(DiffusionPipeline, Flux2LoraLoaderMixin)`,
`model_cpu_offload_seq = "text_encoder->transformer->vae"`.

## Le mapping

```
QwenImagePipeline           -> Flux2KleinPipeline
QwenImageEditPlusPipeline   -> Flux2KleinPipeline           <- MÊME objet
QwenImageInpaintPipeline    -> Flux2KleinInpaintPipeline
QwenImageImg2ImgPipeline    -> Flux2KleinInpaintPipeline    <- masque blanc plein
QwenImageTransformer2DModel -> Flux2Transformer2DModel
```

## Ce que ce fork gagne sur l'amont

**1. Un seul modèle au lieu de deux.** `image` du pipeline base accepte déjà une
**liste** de PIL : `generate_omni` (l. 1977) tape `_ensure_base()` au lieu de charger
un second modèle. Tombent avec :

| Élément | Ligne |
|---|---|
| `_load_omni` | 1896-1975 |
| `DEFAULT_OMNI_REPO` | 34 |
| `set_omni_model` | 1267 |
| `list_edit_models` | 1278 |
| `check_omni_available` | 1305 |
| branche `kind == "omni"` de `get_pipe` | 1845-1851 |
| `set_edit_loras` / `edit_speed_choices` / `set_edit_loras_enabled` | 1201, 1224, 1257 |
| `cz_edit_loras.py` (431 l.) | tout le fichier |

~350 lignes en moins dans `cz_pipeline.py`, et le lazy load n'a plus qu'un modèle à
jongler. Les LoRA d'édition deviennent des LoRA ordinaires (`set_loras`, l. 1184).

**2. `image_reference` sur l'inpaint.** Inédit dans la lignée : on inpaint une zone
**en passant une référence personnage**. Directement exploitable par comics2crispz —
redessiner une case en gardant le casting cohérent.

**3. Le single-file remarche.** Repo tagué `diffusion-single-file`, avec
`flux-2-klein-4b.safetensors` à la racine **et** un layout diffusers complet.
Contrairement à krea2, `list_checkpoints()` (l. 1034), `resolve_checkpoint` (1081) et
l'indexation Civitai gardent tout leur sens. `_load_transformer` (1479-1583) se
transpose tel quel.

## Ce qui diverge de l'amont — à ne PAS écraser lors d'un merge

**A. `negative_prompt` n'existe nulle part.** C'est le coût principal du portage.
Tout `_cfg` (l. 313-324) est bâti sur `true_cfg_scale` + `negative_prompt` ; ni l'un
ni l'autre n'existe côté FLUX.2. Seul `negative_prompt_embeds` est exposé.

| Option | Coût | `supports.negative` |
|---|---|---|
| honnête | 0 | `false` + warning quand un spec porte un `negative` |
| propre | ~30 l. dans `_cfg` via `pipe.encode_prompt()` | `true` |

`_qwen_call` (l. 327) dégrade **déjà** gracieusement sur `TypeError` pour exactement
ces deux kwargs : le filet est en place, quelle que soit l'option retenue.

**B. Pas de `strength` sur le pipeline base → pas d'img2img direct.** `image=` y est
un conditionnement *référence/édition* (façon Kontext), pas un départ bruité. Sans
parade, `_refine_whole` (2383), `_refine_tiled` (2422), `process_one` (2482) et
`txt2img_run(upscale=True)` (2578) n'ont pas de cible.

**Parade — c'est ce qui évite de retomber dans le trou de krea2 :**
`Flux2KleinInpaintPipeline` expose `strength`. Un **masque entièrement blanc** = un
img2img exact. Donc `get_pipe("img2img")` (l. 1839) renvoie le pipe *inpaint*, et
`_refine_*` passe `mask_image=<blanc>, strength=denoise`. `padding_mask_crop` en
prime pour le tuilage.

**C. Guidance.** La doc diffusers précise : *« For step-wise distilled models,
`guidance_scale` is ignored. »* klein-4B est distillé à 4 steps → le curseur
« guidance » de l'UI est peut-être **inopérant**. À mesurer AVANT de coder l'option
A « propre » : deux rendus, `guidance_scale=1.0` vs `8.0`, même seed. Si l'image ne
bouge pas, le débat A est tranché d'office (option honnête).

## `cz_protocol.py` — le caps (l. 184-188)

```python
"supports": {"loras": True,
             "refs": True,          # plus de _omni_configured() : natif au pipeline
             "max_refs": MAX_REFS,
             "seed": True,
             "negative": <selon A>,
             "arbitrary_size": True,
             "edit": True, "inpaint": True, "img2img": True}
```

Toutes les ops du protocole v1 (`caps` / `gen` / `edit` / `inpaint` / `upscale`)
restent servies. **Aucun `exit 3`**, contrairement à crispz-krea2.

Côté comics2crispz, ajouter dans `config.json` :

```json
"klein": { "url": "http://127.0.0.1:7860",
           "czp": "D:/Github/crispz-klein/czp.bat" }
```

## Ordre de portage

1. Test guidance (point C) — décide le point A.
2. `_ensure_base` (1753-1800) + `_load_transformer` (1479-1583) → Flux2.
3. Supprimer la branche omni, recâbler `generate_omni` (1977) sur le base.
4. `get_pipe` (1839) : `img2img` **et** `inpaint` → `Flux2KleinInpaintPipeline`,
   masque blanc plein pour l'img2img.
5. `_cfg` / `_qwen_call` (313-340) selon A.
6. `cz_protocol.py` caps (184-188).
7. `cz_ui.py` : **rien à masquer**, toutes les capacités sont supportées — c'est le
   seul fork de la lignée où `HAS_IMG2IMG` / `HAS_INPAINT` / `HAS_OMNI` sont tous
   vrais.

## Workflow de merge

```bash
git fetch qwen     && git merge qwen/main          # le frère le plus proche
git fetch upstream && git merge upstream/main      # améliorations génériques
```

| Fichier | Stratégie |
|---|---|
| `cz_pipeline.py` | **ours** + porter à la main les améliorations génériques |
| `cz_ui.py`, `cz_core.py`, `config-sample.txt` | **theirs** + réappliquer le delta du fork |

À porter depuis l'amont : `_load_monitor` / `_fmt_load` / `_load_pct`, `_apply_loras`
+ `_APPLIED_LORAS`, `_lora_weight_range`, `_LAST_SEED` / `_NO_SEED_INCREMENT` /
`_SAVE_PRE_UPSCALE`, job queue, XYZ grid, asset browser.

À **ne pas** porter : tout `QwenImage*` / `ZImage*` / `Krea2*`, `true_cfg_scale`,
le second modèle d'édition (`zimage_omni_model`, `zimage_omni_base`) et
`cz_edit_loras.py`.

## Piège du clonage entre forks

Ce fork a été créé par `git clone` (et non par copie de dossier) : `config.txt` et
`preferences.json`, gitignorés, **ne sont pas venus**. C'est délibéré — c'est ce qui
avait coûté quatre échecs de validation à crispz-krea2.

Au premier `cp config-sample.txt config.txt`, vérifier :
`zimage_model` (→ `black-forest-labs/FLUX.2-klein-4B`), `zimage_transformer` (doit
rester absent), `zimage_omni_model` / `zimage_omni_base` (à **supprimer**),
`model_profiles` (les clés `rapid` / `lightning` / `qwen` ne matchent plus rien —
prévoir une clé `klein` à 4 steps), `default_gen_steps` (24 → 4),
`default_guidance` (4.0 → selon le point C), `default_performance`.

## Checklist post-merge

```bash
.venv\Scripts\python -m py_compile app.py cz_pipeline.py cz_ui.py cz_core.py
.venv\Scripts\python -c "import cz_ui; cz_ui.build_ui()"
.venv\Scripts\python -c "import cz_pipeline as p; assert p.round_to_multiple(100)==96"
.venv\Scripts\python -c "import diffusers; diffusers.Flux2KleinPipeline, diffusers.Flux2KleinInpaintPipeline"
```

Puis une génération réelle : `cz_pipeline.generate(prompt=..., width=1024,
height=1024, steps=4, seed=7)`. Baseline à établir au premier run — aucune mesure
n'existe encore pour ce fork.

## Licence

`FLUX.2-klein-4B` est sous **Apache 2.0** : usage commercial libre, pas de plafond de
chiffre d'affaires, pas de filtrage de contenu imposé, pas de révocation. C'est la
licence la plus permissive de toute la lignée crispz — l'inverse exact de crispz-krea2.

⚠️ **Ne pas confondre avec `FLUX.2-klein-9B`**, sous licence **non commerciale** et
imposant des filtres de contenu. Ce fork cible le **4B** ; ne pas basculer
`DEFAULT_BASE_REPO` sur le 9B sans revoir `LICENSE.txt` et `NOTICE`.

## État

- [ ] Test guidance (point C) → tranche le point A.
- [ ] `cz_pipeline.py` : `_ensure_base` + `_load_transformer` → `Flux2KleinPipeline` /
      `Flux2Transformer2DModel`.
- [ ] Suppression de la branche omni (`_load_omni`, `set_omni_model`,
      `list_edit_models`, `check_omni_available`, `cz_edit_loras.py`) ;
      `generate_omni` rebranché sur le pipeline base.
- [ ] `get_pipe` : `img2img` + `inpaint` → `Flux2KleinInpaintPipeline` (masque blanc
      plein pour l'img2img).
- [ ] `_cfg` / `_qwen_call` : sortie de `true_cfg_scale`, décision negative (A).
- [ ] `cz_protocol.py` : caps (`refs: True` natif, `negative` selon A).
- [ ] `config-sample.txt` + `cz_core.py` : défauts klein (4 steps, profil `klein`,
      suppression des clés omni).
- [ ] `requirements.txt` : version diffusers minimale exposant `flux2`. À confirmer.
- [ ] Test génération + édition multi-réf réels sur GPU.
- [ ] Entrée `klein` dans `comics2crispz/config.json`.
- [ ] README + identité (titres, captures).
- [ ] Launcher Pinokio `crispz-klein.pinokio.git`.
