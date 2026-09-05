# crispz-klein — fork FLUX.2 Klein de crispz-qwen-edit

Fork **texte → image + édition multi-référence** basé sur
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
(4B, Apache 2.0, distillé 4 steps). Parti de **crispz-qwen-edit** plutôt que de
crispz-studio ou crispz-krea2, parce que klein-4B a le même profil fonctionnel que
Qwen-Image-Edit — txt2img, img2img, inpaint et édition multi-référence dans un seul
fork — là où krea2 est le fork amputé (`img2img: False`, `inpaint: False`, pas
d'omni, pas de single-file).

- Modèle : `black-forest-labs/FLUX.2-klein-4B` (défaut). **Repo public, Apache 2.0.**
- Remotes : `origin` = crispz-klein, `upstream` = crispz-studio, `qwen` = crispz-qwen-edit.
- Base du fork : `3c128c8` (crispz-qwen-edit/main).

> **État : portage fait, installé et validé de bout en bout** (RTX 5090, torch
> 2.8.0+cu128, diffusers 0.39.0.dev0, 2026-09-05). Le fork a son propre `.venv`
> (`install.bat`), les 4 ops du protocole v1 tournent, la suite héritée passe en
> entier, et comics2crispz atteint le moteur par la route CLI. Reste le launcher
> Pinokio (cf. § État).

## Mesures (RTX 5090, 1024×1024, 4 steps, bf16, offload none)

| Étape | Temps | VRAM |
|---|---|---|
| chargement du modèle | 8,9 s | 14,94 Go |
| `gen` (txt2img) | 2,0 s | — |
| `edit` 1 référence | 3,0 s | — |
| `edit` 2 références | 5,2 s | — |
| `inpaint` | 3,0 s | — |
| `upscale` factor 1 (img2img) | 1,7 s | — |

**14,94 Go au total, quel que soit le nombre de pipelines dérivés** : un seul modèle
sert txt2img, edit, inpaint et img2img (vérifié par `test_klein_e2e.py` étape 6).
Pour mémoire, crispz-qwen-edit charge deux modèles de 20B pour la même surface.

## Pourquoi klein-4B plutôt que Krea 2 ou Anima

| | Qwen-Image-Edit-2511 | **klein-4B** | Krea 2 | Anima turbo |
|---|---|---|---|---|
| Params | 20B | **4B** | 12,9B | 2B |
| VRAM mesurée | ~40 Go (2 modèles) | **14,9 Go** | ~26 Go | ~5 Go |
| Steps | ~8 (Lightning) | **4** | 8 | 8–12 |
| `from_single_file` | oui | **oui, testé** | non | n/a (ComfyUI) |
| img2img / inpaint | oui | oui | **non** | non |
| multi-référence | modèle séparé | **natif, même pipeline** | non | LoRA expérimental |
| Licence | Apache 2.0 | **Apache 2.0** | Community (plafond 1 M$) | — |

Rendu BD : klein sort des cases encrées à plat, avec cadre, sans LoRA de style
(cf. `tests/e2e_1_gen.png`). L'édition préserve le personnage au pixel près tout en
changeant la scène (`tests/e2e_2_edit1.png`, nuit → jour) — c'est précisément ce dont
le casting `@Nom` de comics2crispz a besoin.

## Le mapping

```
QwenImagePipeline           -> Flux2KleinPipeline
QwenImageEditPlusPipeline   -> Flux2KleinPipeline           <- MÊME objet
QwenImageInpaintPipeline    -> Flux2KleinInpaintPipeline
QwenImageImg2ImgPipeline    -> Flux2KleinInpaintPipeline    <- masque blanc plein
QwenImageTransformer2DModel -> Flux2Transformer2DModel
```

Signatures relevées sur `diffusers/main` :

| kwarg | `Flux2KleinPipeline` | `Flux2KleinInpaintPipeline` |
|---|---|---|
| `image` | **`list[PIL] \| PIL`** | oui |
| `image_reference` | — | oui |
| `mask_image` | — | oui |
| `strength` | **absent** | oui (défaut 0.8) |
| `padding_mask_crop` | — | oui |
| `guidance_scale` | 4.0 | 8.0 |
| `negative_prompt` | **absent** | **absent** |

## Ce qui diverge de l'amont — à ne PAS écraser lors d'un merge

### A. Aucun negative prompt, aucun CFG — **tranché par la mesure**

`tests/test_klein_guidance.py` rend la même seed à `guidance_scale` 1.0 / 4.0 / 8.0 :
**images bit-à-bit identiques** (MAE 0,0000, écart max 0). diffusers l'annonce
lui-même — `pipeline_flux2_klein.py:585` : `if guidance_scale > 1.0 and
self.config.is_distilled: logger.warning("Guidance scale ... is ignored")`. Le
`model_index.json` du repo porte `"is_distilled": true`.

Donc **option honnête** retenue (pas de `negative_prompt_embeds` à coder) :

- `_cfg()` renvoie `{}` — la signature est conservée, tous les callsites
  `**_cfg(negative)` de l'amont restent valides ;
- `_qwen_call()` injecte `guidance_scale = 1.0` quand `pipe.config.is_distilled`,
  ce qui **tait le warning diffusers à chaque appel**. Si un checkpoint FLUX.2 NON
  distillé est chargé un jour, `is_distilled` est faux et le curseur de l'UI
  redevient un vrai CFG — le code gère déjà les deux ;
- `cz_protocol` annonce `supports.negative: false` et **émet un warning** quand un
  spec porte un `negative` ou un `guidance` (règle maison : dégradation annoncée).

`GUIDANCE` et le curseur de l'UI survivent (contrat `cz_ui` / `cz_cli`) mais
n'ont aucun effet. `default_guidance` est passé à 1.0 partout pour que l'UI
n'affiche pas une valeur suggérant un CFG actif.

### B. Pas de `strength` sur le pipeline base → img2img via l'inpaint

`image=` du pipeline base est un conditionnement *référence/édition* (façon Kontext),
pas un départ bruité. `Flux2KleinInpaintPipeline` expose `strength` : l'img2img passe
donc par lui avec un **masque entièrement blanc**.

L'injection est faite **dans `_qwen_call`**, pas aux quatre callsites : un appel qui
porte `image` + `strength` sans `mask_image` reçoit un masque blanc à la taille de
l'image. `_refine_whole`, `_refine_tiled`, `process_one` et `txt2img_run(upscale=True)`
sont donc **inchangés** — surface de conflit minimale au merge.

`get_pipe("img2img")` et `get_pipe("inpaint")` renvoient **le même objet** (partagé
sous les deux clefs de `_DERIVED`).

### C. Le catalogue de LoRA d'édition est VIDE — *imprévu, trouvé au test*

`cz_edit_loras.py` héritait de 19 presets Qwen-Image-Edit 2509/2511 et de 2 presets
Lightning. Tous **incompatibles FLUX.2** (architecture et clés différentes). Le `caps`
les annonçait : un appelant demandant `Manga-Tone` aurait planté. `EDIT_LORA_SPECS`
et `SPEED_SPECS` sont donc vidés — **le catalogue amont est conservé juste en dessous,
commenté, comme référence de merge**. Toute la mécanique (téléchargement paresseux,
index local, overrides config) est intacte : une entrée FLUX.2 suffit à la réveiller.

Conséquences dans le `caps` : `edit_loras: []`, `edit_presets: false`,
`edit_fast: ["off"]`. `speed_names()` renvoie `[]` — klein est déjà distillé à
4 steps, il n'y a rien à accélérer.

### D. `_QWEN_KEY_MARKERS` re-dérivé — *imprévu*

La garde d'architecture du chemin single-file/GGUF cherchait `img_in`, `txt_in`,
`time_text_embed` : **aucun n'existe** dans le transformer FLUX.2 (169 tenseurs
relevés). Nouveaux marqueurs : `single_transformer_blocks.`,
`double_stream_modulation`, `x_embedder`, `context_embedder`. Le nom de la constante
est gardé pour limiter la surface de conflit ; seul le contenu change.

### E. `requirements-lock.txt` : `gguf` et `hf_xet` manquaient — *imprévu*

Le lock hérité de crispz-studio omet explicitement `gguf` (« crispz-studio n'a pas de
chemin de chargement GGUF »). C'est faux pour ce fork : `_load_transformer` a bien un
chemin GGUF, hérité de qwen-edit et re-validé ici. Un `install.bat` **isolé** (qui lit
le lock, pas `requirements.txt`) sortait donc un venv sans `gguf` → `test_quant_formats`
en échec et un `.gguf` refusé à l'exécution. `hf_xet` manquait aussi (téléchargements HF
en HTTP lent). Les deux sont ajoutés au lock avec leur version installée.

### F. L'API omni survit, elle ne disparaît PAS

**Correction du plan initial** : `cz_ui.py` et `cz_protocol.py` référencent ces
symboles 20+ fois. Les supprimer casserait les deux. Ils sont donc **repointés**,
pas retirés :

| Symbole | Nouveau comportement |
|---|---|
| `OMNI_MODEL` | suit `BASE_REPO` (toujours non vide → edit dispo) |
| `set_omni_model()` | no-op journalisé (changer le checkpoint change l'éditeur) |
| `list_edit_models()` | renvoie `list_checkpoints()` |
| `check_omni_available()` | message « native », sans appel réseau |
| `get_pipe("omni")` | renvoie `_ensure_base()` |
| `generate_omni()` | tape le pipeline de base, `image=` liste de PIL |
| `_load_omni()` | **seule suppression réelle** (81 lignes) |

## `cz_protocol.py` — le caps effectif

```json
"tool": "crispz-klein",
"ops": ["caps", "gen", "upscale", "edit", "inpaint"],
"supports": {"loras": true, "refs": true, "max_refs": 4, "seed": true,
             "negative": false, "arbitrary_size": true, "faces": true,
             "detail_faces": true, "detail_hands": true,
             "edit": true, "edit_presets": false,
             "inpaint": true, "img2img": true}
```

**Aucun `exit 3`** : les 4 ops sont servies, y compris `upscale` factor 1
(variation img2img) que crispz-krea2 refuse.

Côté comics2crispz, ajouter dans `config.json` :

```json
"klein": { "url": "http://127.0.0.1:7860",
           "czp": "D:/Github/crispz-klein/czp.bat" }
```

## Workflow de merge

```bash
git fetch qwen     && git merge qwen/main          # le frère le plus proche
git fetch upstream && git merge upstream/main      # améliorations génériques
```

| Fichier | Stratégie |
|---|---|
| `cz_pipeline.py` | **ours** + porter à la main les améliorations génériques |
| `cz_edit_loras.py` | **ours** (le catalogue amont doit rester commenté) |
| `cz_ui.py`, `cz_core.py`, `config-sample.txt` | **theirs** + réappliquer le delta du fork |

À porter depuis l'amont : `_load_monitor` / `_fmt_load` / `_load_pct`, `_apply_loras`
+ `_APPLIED_LORAS`, `_lora_weight_range`, `_LAST_SEED` / `_NO_SEED_INCREMENT` /
`_SAVE_PRE_UPSCALE`, job queue, XYZ grid, asset browser.

À **ne pas** porter : tout `QwenImage*` / `ZImage*` / `Krea2*`, `true_cfg_scale`,
les clés `zimage_omni_model` / `zimage_omni_base`, le catalogue de LoRA d'édition
Qwen, les presets Lightning, les marqueurs de clés Qwen.

## Piège du clonage entre forks

Ce fork a été créé par `git clone` (et non par copie de dossier) : `config.txt` et
`preferences.json`, gitignorés, **ne sont pas venus**. C'est délibéré — c'est ce qui
avait coûté quatre échecs de validation à crispz-krea2.

Au premier `cp config-sample.txt config.txt`, vérifier : `zimage_model`
(→ `black-forest-labs/FLUX.2-klein-4B`), `zimage_transformer` (doit rester absent),
`zimage_omni_model` / `zimage_omni_base` (**supprimées du sample, ne pas les
réintroduire**), `model_profiles` (clés `klein` / `flux-2` / `flux2` à 4 steps),
`default_gen_steps` (4), `default_guidance` (1.0), `default_performance`
(`Turbo (4 steps)`), `default_cpu_offload` (`none` : klein tient en 15 Go, contre
`model` pour les 20B de Qwen).

## Tests

```bash
.venv\Scripts\python tests\test_klein_guidance.py   # point A/C : guidance ignoré
.venv\Scripts\python tests\test_klein_e2e.py        # les 4 ops + partage des pipes
```

Suite héritée : **13/13 au vert**. Deux fixtures ont dû être portées dans
`test_quant_formats.py` — les clés synthétiques (marqueurs Qwen) et la classe
monkeypatchée (`QwenImageTransformer2DModel` → `Flux2Transformer2DModel`).

Smoke test du protocole :

```bash
.venv\Scripts\python cz_protocol.py caps
.venv\Scripts\python cz_protocol.py gen --spec spec.json --local
```

## Licence

`FLUX.2-klein-4B` est sous **Apache 2.0** : usage commercial libre, pas de plafond de
chiffre d'affaires, pas de filtrage de contenu imposé, pas de révocation. C'est la
licence la plus permissive de toute la lignée crispz — l'inverse exact de crispz-krea2.

⚠️ **Ne pas confondre avec `FLUX.2-klein-9B`**, sous licence **non commerciale** et
imposant des filtres de contenu. Ce fork cible le **4B** ; ne pas basculer
`DEFAULT_BASE_REPO` sur le 9B sans revoir `LICENSE.txt` et `NOTICE`.

## État

- [x] Test guidance (point C) → guidance ignoré, images bit-à-bit identiques.
- [x] `cz_pipeline.py` : `_ensure_base` + `_load_transformer` → `Flux2KleinPipeline` /
      `Flux2Transformer2DModel`.
- [x] `_load_omni` supprimé ; `generate_omni` et `get_pipe("omni")` sur le base ;
      API omni repointée (cf. § E).
- [x] `get_pipe` : `img2img` + `inpaint` → `Flux2KleinInpaintPipeline`, objet partagé,
      masque blanc injecté dans `_qwen_call`.
- [x] `_cfg` / `_qwen_call` : sortie de `true_cfg_scale`, `guidance_scale=1.0` si distillé.
- [x] `cz_protocol.py` : caps + warnings `negative` / `guidance`.
- [x] `cz_edit_loras.py` : catalogue Qwen et presets Lightning vidés (§ C).
- [x] `_QWEN_KEY_MARKERS` re-dérivé sur le vrai transformer FLUX.2 (§ D).
- [x] `config-sample.txt` + `cz_core.py` : défauts klein.
- [x] Chemin single-file testé (checkpoint `.safetensors` de 7,2 Go chargé et rendu).
- [x] Les 4 ops du protocole testées en `--local`.
- [x] Suite héritée 13/13, `build_ui()` headless OK.
- [x] `requirements.txt` : le commit diffusers déjà épinglé (`de6b0495`) expose
      `flux2` en 0.39.0.dev0 ; commentaires corrigés. `install.bat` CHECK_PIPE →
      `Flux2KleinInpaintPipeline` (validé par install.bat lui-même).
- [x] `requirements-lock.txt` : `gguf` + `hf_xet` ajoutés (§ E).
- [x] `.venv` du fork installé (`install.bat`, 8,6 Go) ; `config.txt` généré depuis
      le sample porte bien les défauts klein — le piège du clonage est clos par
      construction.
- [x] Suite héritée 13/13 et e2e revalidés **sur le venv du fork**.
- [x] Entrée `klein` dans `comics2crispz/config.json` ; caps atteint par la route
      CLI (`route: cli`, `crispz-klein 1.18.0`).
- [x] README : en-tête et identité. Le corps garde encore la formulation
      crispz-studio héritée (dette d'amont, pas introduite par ce portage).
- [x] CHANGELOG 1.18.0, `APP_VERSION` bumpée.
- [ ] Launcher Pinokio `crispz-klein.pinokio.git`.
- [ ] README : réécrire le corps (dette héritée de crispz-studio).

## Note d'exploitation

La famille partage le port 7860 (décision v1 : « pas de port par outil »). Tant qu'une
autre app crispz tourne, `caps` par l'URL renvoie CELLE-LÀ — comics2crispz le détecte
(`tool_mismatch: True`) et `upscale`/`gen` refusent proprement. Pour utiliser klein,
fermer l'autre app et lancer `run.bat`, ou passer par la route CLI (`--local`).
