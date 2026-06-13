"""
modelcopyV50YOLO.py — Test d'un détecteur YOLO sur JHU-CROWD++
==============================================================

ARCHITECTURE COMPLÈTEMENT DIFFÉRENTE des V1-V49.

  V1-V49 : U-Net encodeur-décodeur, sortie HEATMAP dense (régression de
           gaussiennes), travaille sur des PAIRES de frames (A→B) pour le
           tracking, supervision par carte de densité.

  V50    : YOLO (You Only Look Once) — détecteur de BOÎTES en une passe.
           Travaille sur une SEULE image, produit une liste de boîtes
           englobantes + scores. Pas de heatmap, pas de paires, pas de
           tracking : c'est de la détection d'objets pure.

YOLO détecte des TÊTES (pas seulement des personnes COCO). Trois cibles
possibles via DETECTION_TARGET (réglée en bas du fichier) :

  "head_finetuned" (défaut) — détecteur de TÊTES obtenu en fine-tunant YOLO
      sur les boîtes de têtes JHU (voir MODE="train"). C'est la voie la plus
      fiable pour avoir des têtes sur JHU sans dépendre de poids externes.

  "head_external" — détecteur de TÊTES à partir de poids pré-entraînés
      externes (typiquement entraînés sur CrowdHuman, le dataset de têtes en
      foule). Renseigner HEAD_WEIGHTS avec le chemin du .pt téléchargé.
      Permet de voir des têtes SANS entraîner, si on a les poids.

  "person_coco" — YOLO pré-entraîné COCO, classe « person » (corps entier).
      Conservé comme BASELINE de comparaison (ce que faisait V50 au départ).

Pourquoi « le résultat sera probablement faux » (attendu) :
  - JHU-CROWD++ est dense (jusqu'à >10 000 têtes/image). Les détecteurs à
    boîtes s'effondrent en foule dense : la NMS supprime aussi les vraies
    détections voisines. Les têtes lointaines font quelques pixels.
  - person_coco détecte le corps entier alors que JHU annote des têtes
    → gros sous-comptage. Les cibles « head_* » corrigent ce point mais
    restent limitées sur les très grandes foules.

Modes (variable MODE, en bas du fichier) :
  "predict" (défaut) — inférence + visualisation + comparaison comptage GT.
  "train"            — convertit JHU au format YOLO et fine-tune sur les têtes
                       (produit head_finetuned).
  "compare"          — sur les mêmes images, juxtapose person_coco vs têtes.

Dépendance : ultralytics  (pip install ultralytics)
"""

import os
import random

import cv2
import numpy as np

# --- Réglages généraux -------------------------------------------------------

# Cible de détection : "head_finetuned" | "head_external" | "person_coco"
DETECTION_TARGET = "head_finetuned"

# Poids YOLO COCO de base (téléchargé automatiquement par ultralytics).
# Sert de point de départ au fine-tuning ET de baseline "person_coco".
PERSON_WEIGHTS = "yolo11n.pt"

# Poids du détecteur de têtes fine-tuné sur JHU (produit par MODE="train").
HEAD_FINETUNED_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "runs", "detect", "v50yolo_heads", "weights", "best.pt"
)

# Poids de têtes EXTERNES (ex. YOLO entraîné sur CrowdHuman). À télécharger
# soi-même puis renseigner ici le chemin local du .pt.
HEAD_WEIGHTS = os.path.join(os.path.dirname(__file__), "crowdhuman_head.pt")

IMG_SIZE   = 1280   # taille d'inférence/entraînement (têtes petites → grand imgsz)
CONF_THRES = 0.25   # seuil de confiance des détections
MAX_DET    = 10000  # nb max de boîtes par image (foule dense → relever fortement)


def resolve_target(target: str) -> tuple[str, int, str]:
    """
    Traduit une cible de détection en (poids, classe ciblée, libellé).

    - head_finetuned : poids JHU fine-tunés, classe 0 = head.
    - head_external  : poids CrowdHuman/autres, classe « head » détectée
                       automatiquement dans les noms du modèle (sinon 0).
    - person_coco    : poids COCO, classe 0 = person.

    Returns : (weights_path, target_class, libellé court).
    """
    if target == "head_finetuned":
        return HEAD_FINETUNED_WEIGHTS, 0, "tete"
    if target == "head_external":
        return HEAD_WEIGHTS, None, "tete"   # classe résolue à l'inférence
    if target == "person_coco":
        return PERSON_WEIGHTS, 0, "personne"
    raise ValueError(f"DETECTION_TARGET inconnu : {target!r}")


# =============================================================================
# Lecture des annotations JHU-CROWD++
# =============================================================================

def _read_gt(gt_path: str) -> list[tuple[int, int, int, int]]:
    """
    Lit un fichier d'annotations JHU (gt/*.txt).

    Format d'une ligne : x y w h occlusion blur
        (x, y) = centre de la tête en pixels ; (w, h) = taille de la boîte.

    Returns : liste de (x, y, w, h) en pixels.
    """
    boxes = []
    if not os.path.exists(gt_path):
        return boxes
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, w, h = (int(float(parts[i])) for i in range(4))
            boxes.append((x, y, w, h))
    return boxes


def _load_counts(data_root: str, split: str) -> dict[str, int]:
    """Lit image_labels.txt → {id: comptage de personnes attendu}."""
    counts = {}
    path = os.path.join(data_root, split, "image_labels.txt")
    with open(path, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) >= 2:
                counts[parts[0]] = int(parts[1])
    return counts


# =============================================================================
# Conversion JHU → format YOLO  (pour le fine-tuning sur les TÊTES)
# =============================================================================

def convert_split_to_yolo(data_root: str, split: str) -> int:
    """
    Convertit les annotations d'un split au format YOLO.

    Pour chaque image data/<split>/images/<id>.jpg, écrit
    data/<split>/labels/<id>.txt avec, par ligne :
        <classe> <cx> <cy> <w> <h>     (coordonnées NORMALISÉES dans [0, 1])
    classe = 0 (tête, classe unique). ultralytics retrouve automatiquement
    ces labels en remplaçant « /images/ » par « /labels/ » dans le chemin.

    Non destructif : ajoute seulement un dossier labels/ à côté de images/.

    Returns : nombre d'images converties.
    """
    images_dir = os.path.join(data_root, split, "images")
    gt_dir     = os.path.join(data_root, split, "gt")
    labels_dir = os.path.join(data_root, split, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    counts = _load_counts(data_root, split)
    n_done = 0
    for img_id in counts:
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        lines = []
        for (x, y, w, h) in _read_gt(os.path.join(gt_dir, f"{img_id}.txt")):
            w = max(w, 1)
            h = max(h, 1)
            cx, cy = x / W, y / H
            nw, nh = w / W, h / H
            # clamp dans [0, 1] (sécurité bordures)
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            nw, nh = min(nw, 1.0), min(nh, 1.0)
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(os.path.join(labels_dir, f"{img_id}.txt"), "w") as f:
            f.write("\n".join(lines))
        n_done += 1

    print(f"  [{split}] {n_done} images converties → {labels_dir}")
    return n_done


def make_data_yaml(data_root: str, yaml_path: str) -> str:
    """Génère le data.yaml ultralytics (classe unique : head)."""
    data_root_abs = os.path.abspath(data_root).replace("\\", "/")
    content = (
        f"path: {data_root_abs}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"names:\n"
        f"  0: head\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  data.yaml écrit : {yaml_path}")
    return yaml_path


# =============================================================================
# Entraînement (fine-tuning YOLO sur les têtes)
# =============================================================================

def train_yolo(
    data_root: str,
    weights: str = PERSON_WEIGHTS,
    epochs: int = 50,
    img_size: int = IMG_SIZE,
    batch: int = 8,
) -> None:
    """
    Fine-tune YOLO sur la détection de têtes JHU.

    Étapes :
      1. Convertit train/ et val/ au format YOLO (labels/).
      2. Génère data.yaml.
      3. Lance ultralytics YOLO(weights).train(...).

    Les poids entraînés sont sauvegardés par ultralytics dans runs/detect/.../weights/best.pt
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERREUR] ultralytics introuvable. Installez-le :  pip install ultralytics")
        return

    print("== Conversion des annotations au format YOLO ==")
    convert_split_to_yolo(data_root, "train")
    convert_split_to_yolo(data_root, "val")

    yaml_path = os.path.join(os.path.dirname(__file__), "jhu_head.yaml")
    make_data_yaml(data_root, yaml_path)

    print(f"== Fine-tuning YOLO ({weights}) : {epochs} époques, imgsz={img_size} ==")
    model = YOLO(weights)
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        name="v50yolo_heads",
    )
    print("Entraînement terminé. Poids : runs/detect/v50yolo_heads/weights/best.pt")


# =============================================================================
# Inférence + visualisation
# =============================================================================

def _load_model(weights: str):
    """
    Charge un modèle YOLO. Pour des poids locaux (.pt fine-tuné ou externe),
    vérifie l'existence du fichier et renvoie un message clair sinon.
    Les noms courts type "yolo11n.pt" sont téléchargés par ultralytics.

    Returns : modèle YOLO, ou None si indisponible.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERREUR] ultralytics introuvable. Installez-le :  pip install ultralytics")
        return None

    is_local = weights.endswith(".pt") and ("/" in weights or "\\" in weights)
    if is_local and not os.path.isfile(weights):
        print(f"[ERREUR] Poids introuvables : {weights}")
        print("  → Pour des TÊTES, deux options :")
        print("     1. MODE='train' : fine-tune YOLO sur les têtes JHU (produit best.pt).")
        print("     2. DETECTION_TARGET='head_external' + télécharger des poids CrowdHuman,")
        print("        puis renseigner HEAD_WEIGHTS avec leur chemin.")
        print("  (ou DETECTION_TARGET='person_coco' pour la baseline personnes.)")
        return None
    return YOLO(weights)


def _resolve_class(model, target_class: int | None) -> tuple[int | None, str]:
    """
    Détermine la classe à détecter et son nom.

    Si target_class est None (poids externes), cherche une classe nommée
    « head » parmi les noms du modèle ; à défaut, prend la classe 0.

    Returns : (id de classe, nom de classe).
    """
    names = model.names  # dict {id: nom}
    if target_class is None:
        for cid, cname in names.items():
            if "head" in str(cname).lower():
                return cid, cname
        return 0, names.get(0, "0")
    return target_class, names.get(target_class, str(target_class))


def _detect_boxes(model, img, cls_id, conf, img_size):
    """Inférence YOLO sur une image → tableau (N, 4) de boîtes xyxy."""
    classes = [cls_id] if cls_id is not None else None
    res = model.predict(
        img, conf=conf, imgsz=img_size,
        classes=classes, max_det=MAX_DET, verbose=False,
    )[0]
    if res.boxes is None:
        return np.empty((0, 4))
    return res.boxes.xyxy.cpu().numpy()


def predict_and_visualize(
    target: str,
    data_root: str,
    output_dir: str,
    split: str = "test",
    n_samples: int = 8,
    conf: float = CONF_THRES,
    img_size: int = IMG_SIZE,
    indices: list[int] | None = None,
) -> None:
    """
    Charge le détecteur correspondant à `target` (têtes ou personnes), détecte
    sur des images de test, dessine les boîtes et compare le comptage détecté
    au comptage réel (image_labels.txt). Une image annotée par échantillon est
    sauvegardée dans output_dir.
    """
    weights, target_class, label = resolve_target(target)
    model = _load_model(weights)
    if model is None:
        return

    cls_id, cls_name = _resolve_class(model, target_class)
    print(f"Cible : {target}  |  poids : {weights}")
    print(f"Classe détectée : {cls_id} ({cls_name}) → comptée comme « {label} »")

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(data_root, split, "images")
    counts     = _load_counts(data_root, split)
    img_ids    = list(counts.keys())

    if indices is not None:
        selected = [img_ids[i] for i in indices if 0 <= i < len(img_ids)]
    else:
        selected = random.sample(img_ids, min(n_samples, len(img_ids)))

    print(f"Inférence sur {len(selected)} images (split={split})")

    for k, img_id in enumerate(selected, 1):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [SKIP] image illisible : {img_path}")
            continue

        boxes  = _detect_boxes(model, img, cls_id, conf, img_size)
        n_pred = len(boxes)
        n_gt   = counts.get(img_id, "?")

        vis = img.copy()
        for (x1, y1, x2, y2) in boxes.astype(int):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

        bar_h = 40
        bar = np.zeros((bar_h, vis.shape[1], 3), dtype=np.uint8)
        txt = f"id={img_id}  GT={n_gt}  YOLO[{label}]={n_pred} det.  conf>={conf}"
        cv2.putText(bar, txt, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out = np.vstack([bar, vis])

        out_path = os.path.join(output_dir, f"sample_{img_id}.png")
        cv2.imwrite(out_path, out)
        print(f"  [{k}/{len(selected)}] id={img_id}  GT={n_gt}  YOLO[{label}]={n_pred}  → {out_path}")

    print(f"\nVisualisations sauvegardées dans : {output_dir}")


def compare_targets(
    data_root: str,
    output_dir: str,
    split: str = "test",
    n_samples: int = 8,
    conf: float = CONF_THRES,
    img_size: int = IMG_SIZE,
) -> None:
    """
    Juxtapose, sur les mêmes images, la détection « personnes COCO » (gauche)
    et la détection « têtes » (droite, DETECTION_TARGET courant si head_*).
    Illustre concrètement la différence personnes vs têtes.
    """
    head_target = DETECTION_TARGET if DETECTION_TARGET.startswith("head") else "head_finetuned"
    w_person, c_person, _ = resolve_target("person_coco")
    w_head,   c_head,   _ = resolve_target(head_target)

    m_person = _load_model(w_person)
    m_head   = _load_model(w_head)
    if m_person is None or m_head is None:
        return
    cid_p, name_p = _resolve_class(m_person, c_person)
    cid_h, name_h = _resolve_class(m_head, c_head)

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(data_root, split, "images")
    counts     = _load_counts(data_root, split)
    selected   = random.sample(list(counts.keys()), min(n_samples, len(counts)))
    print(f"Comparaison personnes ({name_p}) vs têtes ({name_h}) sur {len(selected)} images")

    for k, img_id in enumerate(selected, 1):
        img = cv2.imread(os.path.join(images_dir, f"{img_id}.jpg"))
        if img is None:
            continue
        n_gt = counts.get(img_id, "?")

        panels = []
        for model, cid, label, color in [
            (m_person, cid_p, "personnes", (0, 200, 255)),
            (m_head,   cid_h, "tetes",     (0, 255, 0)),
        ]:
            boxes = _detect_boxes(model, img, cid, conf, img_size)
            p = img.copy()
            for (x1, y1, x2, y2) in boxes.astype(int):
                cv2.rectangle(p, (x1, y1), (x2, y2), color, 1)
            bar = np.zeros((40, p.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, f"{label}={len(boxes)}  (GT={n_gt})", (8, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            panels.append(np.vstack([bar, p]))

        out = np.concatenate(panels, axis=1)
        out_path = os.path.join(output_dir, f"compare_{img_id}.png")
        cv2.imwrite(out_path, out)
        print(f"  [{k}/{len(selected)}] id={img_id} → {out_path}")

    print(f"\nComparaisons sauvegardées dans : {output_dir}")


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    DATA_ROOT = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
    VIZ_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "visualizations", "50YOLO")
    )

    # "predict" → inférence (DETECTION_TARGET) + visualisation + comparaison comptage
    # "train"   → convertit JHU au format YOLO et fine-tune sur les TÊTES
    # "compare" → personnes COCO vs têtes côte à côte sur les mêmes images
    MODE = "train"

    print(f"data/ : {DATA_ROOT}  |  cible : {DETECTION_TARGET}")

    if not os.path.isdir(DATA_ROOT):
        print("[ERREUR] data/ introuvable.")
    elif MODE == "train":
        train_yolo(
            data_root=DATA_ROOT,
            weights=PERSON_WEIGHTS,
            epochs=50,
            img_size=IMG_SIZE,
            batch=8,
        )
    elif MODE == "predict":
        predict_and_visualize(
            target=DETECTION_TARGET,
            data_root=DATA_ROOT,
            output_dir=VIZ_DIR,
            split="test",
            n_samples=8,
            conf=CONF_THRES,
            img_size=IMG_SIZE,
        )
    elif MODE == "compare":
        compare_targets(
            data_root=DATA_ROOT,
            output_dir=VIZ_DIR,
            split="test",
            n_samples=8,
            conf=CONF_THRES,
            img_size=IMG_SIZE,
        )
    else:
        print(f"[ERREUR] MODE inconnu : {MODE!r}")
