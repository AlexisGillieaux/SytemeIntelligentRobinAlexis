# SytemeIntelligentRobinAlexis

Projet réalisé par deux étudiants de l'Hénallux Pierrard (Virton) dans le cadre du
cours de Systèmes Intelligents.

Objectif : à partir du dataset **JHU-CROWD++**, suivre (tracker) les personnes au
sein d'une foule entre deux frames consécutives. Le réseau (`CrowdTrackingNet`, un
U-Net) reçoit une paire **Frame A → Frame B** et apprend à localiser les têtes et à
associer chaque tête de B à sa position d'origine dans A.

> Comme le dataset est composé d'**images fixes**, les paires de frames sont
> **synthétisées** : Frame B est une version géométriquement déformée (warp) de
> Frame A, ce qui simule un léger mouvement de caméra + un déplacement propre à
> chaque tête.

---

## Architecture du dossier

```
SytemeIntelligentRobinAlexis/
├── README.md                    ← ce fichier
├── requirement.txt              ← dépendances Python
├── testcuda.py                  ← vérifie la disponibilité de CUDA
│
├── data/             [non versionné] ← dataset JHU-CROWD++ ORIGINAL (non modifié)
│   ├── train/  val/  test/      ← chaque split : images/ + gt/ + image_labels.txt
│   ├── License  README
│
├── data_tracking/    [non versionné] ← paires de frames PRÉ-GÉNÉRÉES (legacy)
│   └── train/ val/ test/        ← frames_A/ frames_B/ gt_A/ gt_B/ gt_links/
│                                  (désormais générées à la volée, cf. dataGathering)
│
├── data_result/      [non versionné] ← visualisations + labels de tracking (dataResult.py)
│   └── train/ val/ test/        ← viz/ (frames annotées) + labels/ (événements)
│
├── videos/           [non versionné] ← vidéos de test (.mp4) pour videoTest.py
│
├── projects/                    ← CODE ACTIF
│   ├── dataCleaning.py          ← génération des paires + warp géométrique
│   ├── dataGathering.py         ← Dataset/DataLoader PyTorch (paires à la volée)
│   ├── modelcopyV51.py          ← modèle FINAL (U-Net heatmap + offsets)
│   ├── modelcopyV50YOLO.py      ← détecteur YOLO (têtes), architecture alternative
│   ├── videoTest.py             ← test V51 / V50YOLO sur une VIDÉO (rendu annoté)
│   ├── jhu_head.yaml            ← config dataset YOLO (généré)
│   ├── crowd_tracking_net51.pth ← checkpoint du modèle final
│   └── OLD/                     ← ARCHIVES
│       ├── SourceCodes/         ← anciens modelcopyV*.py (V2 → V49)
│       ├── Models/              ← anciens checkpoints (.pth)
│       └── dataResult.py        ← génération des sorties visuelles/labels
│
├── runs/             [non versionné] ← sorties d'entraînement YOLO (ultralytics)
│   └── detect/v50yolo_heads/weights/best.pt
│
├── Output/                      ← courbes d'entraînement (crowd_tracking_netXX_courbes.png)
│
└── visualizations/              ← visus de test par version (visualizations/XX/sample_*.png)
```

> **[non versionné]** : les dossiers `data/`, `data_tracking/`, `data_result/`,
> `videos/` et `runs/` **n'apparaissent pas sur GitHub**. Ce sont des dossiers de
> **données / sorties** (dataset JHU-CROWD++, paires générées, vidéos de test,
> checkpoints YOLO), volumineux et non pertinents pour le versionnement du code
> → ils sont exclus via le `.gitignore`. Il faut donc se procurer le dataset et
> les vidéos séparément pour reproduire l'entraînement et les tests.

### Pipeline

1. **`dataCleaning.py`** — à partir d'une image originale, calcule un warp
   homographique (pan global + jitter par coin) et produit Frame A (crop net) et
   Frame B (crop warpé), avec les annotations et les liens A↔B par tête.
2. **`dataGathering.py`** — `JHUCrowdTrackingDataset` régénère ces paires **à la
   volée** à chaque époque (warp différent → diversité), et fournit le DataLoader.
3. **`modelcopyVXX.py`** — définit le réseau, la perte, l'entraînement, les
   métriques et les visualisations. Deux modes : `train` ou `visualize`.

---

## Historique des modèles

Chaque version part de la précédente et ne change (en général) qu'**un seul
paramètre** pour isoler son effet. Tous utilisent un U-Net (entrée 6 canaux =
Frame A ⊕ Frame B) et l'optimiseur Adam.

### Phase 1 — Mise en place (V1 → V18) · *carte dense mono-canal*

| Ver. | Modification par rapport au précédent |
|------|----------------------------------------|
| V1 | Baseline initiale (point de départ du projet). |
| V2 | Passage à une **architecture U-Net** : sortie dense 1 canal encodant stayed/entered/left (au lieu d'un classifieur global). |
| V3–V5 | Réglages de la perte MSE et des hyper-paramètres d'entraînement. |
| V6–V9 | Exploration d'optimiseurs et de formulations : variantes `ADAM` (AdamW vs Adam) et `trackOLD` (ancienne formulation du tracking) testées en parallèle. |
| V10 (`Gaussian`) | Les cibles deviennent des **gaussiennes** centrées sur les têtes (σ=7) au lieu de points/traits durs → supervision plus douce. |
| V11–V12 | Micro-réglages autour des gaussiennes (σ=7). |
| V13 | Gaussiennes plus **serrées** (σ=5). |
| V14–V17 | Retour σ=7 + ajustements de pondération de la perte. |
| V18 | Gaussiennes très serrées (σ=4). |

### Phase 2 — Tâche simplifiée « stayed » (V19 → V26) · *focus sur les têtes présentes dans les 2 frames*

| Ver. | Modification |
|------|--------------|
| V19 | **Simplification de la tâche** : on ne supervise plus que les têtes « stayed ». Gaussiennes serrées (σ=2) aux deux extrémités ; perte réduite à `w_bg·MSE_fond + w_stayed·MSE_stayed`. |
| V20–V26 | Balayage fin de σ (2↔3 px) et des pondérations `w_bg` / `w_stayed`. |

### Phase 3 — Encodage asymétrique A/B sur 1 canal (V27 → V30)

| Ver. | Modification |
|------|--------------|
| V27 | **Encodage asymétrique** sur un seul canal : Frame B = gaussienne **+2.0**, Frame A = gaussienne **−1.0** (`w_b_peak=20`, `w_a_peak=20`). |
| V28 | `w_b_peak` 20 → **30** (renforce Frame B). |
| V29 | `w_a_peak` 20 → **10** (meilleur recall observé). |
| V30 | Retour `w_b_peak=w_a_peak=20`. |

### Phase 4 — Correction des seuils & montée en résolution (V31 → V35)

| Ver. | Modification |
|------|--------------|
| V31 | **Correction des seuils** de visualisation/métriques (le blanc n'était jamais atteint, le rouge se déclenchait sur tout le fond) : seuils proportionnels à l'amplitude cible. Base = V29. |
| V32–V34 | Ablation de σ (« gaussiennes profondes ») sur la base de V31. |
| V35 | **Résolution 512→1024**, batch 16→4, époques 50→**100**. |

### Phase 5 — Ablation de σ en 1024² (V36 → V39)

| Ver. | Modification |
|------|--------------|
| V36 | Combine V34 (grandes gaussiennes) + V35 (1024², 100 ép.) ; ajoute warmup du LR et `w_a_peak`=`w_b_peak`. σ=**9**. |
| V37 | σ 9 → **12**. |
| V38 | σ 12 → **15** → **meilleur modèle** de la phase. |
| V39 | σ 15 → **18**. |

### Phase 6 — Sortie 2 canaux (V40 → V47) · *Frame A et B séparées*

| Ver. | Modification |
|------|--------------|
| V40 | **Sortie 2 canaux** : Frame B (canal 0) et Frame A (canal 1) sur des canaux distincts, toutes deux en gaussiennes **+2.0** → fini l'écrasement de A par B. Ajout des métriques précision/recall pour A. Base = V38. |
| V41 | `w_a_peak` 20 → **40** (plus de poids sur A). |
| V42 | **Têtes de décodeur séparées** : une branche conv dédiée par canal (head_B / head_A). |
| V43 | σ 15 → **12**. |
| V44 | Base = V42 (têtes séparées) + `w_b_peak`=`w_a_peak`=**30**. |
| V45 | `w_peak` = **40**. |
| V46 | `w_peak` = **50**. |
| V47 | `w_peak` = **60**. |

### Phase 7 — Association & netteté (V48 → V49) · *vers le vrai tracking*

| Ver. | Modification |
|------|--------------|
| V48 | **Tête d'offsets façon CenterTrack** : 2 canaux supplémentaires (Δx, Δy) qui, à chaque pic de Frame B, pointent vers la position de la même tête dans Frame A → **association directe A↔B** (flèches vertes dans les visus). Supervision L1 masquée. Introduit aussi le **warp géométrique** (déplacement variable par tête) dans la génération des paires. Base = V45. |
| V49 | Essai d'une **focal loss (CenterNet)** pour corriger le flou de B → a **dégradé** les performances (sorties sous-confiantes). **Reverté** au régime MSE pondérée éprouvé de V44-V48, en conservant la tête d'offsets et le warp. |

### Phase 8 — Modèle final & architecture alternative (V50 → V51)

| Ver. | Modification |
|------|--------------|
| **V50YOLO** | **Architecture totalement différente** : détecteur de boîtes **YOLO** (ultralytics) au lieu du U-Net à heatmap. Détection image par image (pas de paires, pas de tracking). Détecte des **têtes** (fine-tuné sur JHU) ou des personnes (COCO). Sert de comparaison. |
| **V51** | Modèle **FINAL** : reprend l'architecture U-Net heatmap + offsets (lignée V44-V49 revertée) avec le checkpoint retenu (`crowd_tracking_net51.pth`). |

---

## Lancer un entraînement / une visualisation

**Modèle U-Net (V51)** — le mode se règle en bas du fichier (`MODE = "train"` ou `"visualize"`) :

```bash
cd projects
python modelcopyV51.py
```

Checkpoints à la racine (`crowd_tracking_netXX.pth`), courbes dans `Output/`,
visus de test dans `visualizations/XX/`.

**Détecteur YOLO (V50YOLO)** — `MODE` = `"predict"` / `"train"` / `"compare"`,
cible via `DETECTION_TARGET` :

```bash
cd projects
python modelcopyV50YOLO.py
```

## Test sur une vidéo (rendu final annoté)

`videoTest.py` applique un modèle sur une vidéo et écrit **une seule vidéo
annotée en temps réel**. Le modèle se choisit via `MODEL` (`"V51"` ou
`"V50YOLO"`) ; pour chaque paire de frames consécutives **(i, i+1) = (A, B)** :

- **V51** dessine chaque tête courante + une flèche vers sa position
  précédente (suivi du mouvement, via les offsets) ;
- **V50YOLO** dessine les boîtes détectées sur la frame courante.

```bash
cd projects
python videoTest.py
```

Entrée : `videos/source/<vidéo>.mp4` → sortie : `videos/trained/annotated_<MODEL>_<vidéo>.mp4`.
