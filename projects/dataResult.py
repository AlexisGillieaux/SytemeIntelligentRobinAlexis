"""
dataResult.py — Génération des sorties visuelles et labels d'entraînement
==========================================================================

Ce script lit les paires de frames générées par ``dataCleaning.py``
(stockées dans ``data_tracking/``) et produit, pour chaque paire, deux
fichiers prêts à être utilisés pour entraîner le modèle de tracking :

  1. IMAGE DE VISUALISATION (``viz/id.jpg``)
     Frame B annotée qui montre visuellement ce qu'il se passe entre les deux frames :

       — Trait NOIR   : tête « stayed » (présente dans les deux frames)
            - Relie la position de la tête dans Frame A (en coords Frame B)
              à sa position dans Frame B.
       ● Point NOIR (+1) : tête « entered » (apparue dans Frame B, absente de Frame A)
       × Croix NOIRE (-1): tête « left »    (absente de Frame B, était dans Frame A)
            - Dessinée à sa position de sortie, clampée à la bordure de Frame B.

  2. FICHIER DE LABELS NUMÉRIQUES (``labels/id.txt``)
     Un fichier texte, une ligne par événement de tracking, au format :

         status  x_B  y_B  delta_x  delta_y

     Champs :
       status   :  0 = stayed   → tête présente dans les deux frames
                  +1 = entered  → tête nouvelle dans Frame B
                  -1 = left     → tête qui a quitté le champ
       (x_B, y_B)      : position dans l'espace pixels de Frame B
                          (ou position clampée à la bordure pour les "left")
       (delta_x, delta_y) : déplacement de Frame A → Frame B
                             stayed  → (x_B - x_A, y_B - y_A) = (-dx_global, -dy_global)
                             entered → (0, 0)  pas d'historique Frame A
                             left    → vecteur vers le point de sortie sur la bordure

Logique de coordonnées
----------------------
Les deux frames sont des crops d'une même image originale décalés de (dx, dy) :
    Frame A = original[0 : H-dy, 0 : W-dx]      (coin supérieur-gauche)
    Frame B = original[dy : H,   dx : W]          (coin inférieur-droit)

Pour une tête « stayed » à la position absolue (X, Y) :
    • Dans Frame A (espace pixel) : (x_A, y_A) = (X,      Y     )
    • Dans Frame B (espace pixel) : (x_B, y_B) = (X - dx, Y - dy)
    • Relation directe : x_A = x_B + dx,  y_A = y_B + dy

La flèche dessinée sur Frame B va donc de (x_B + dx, y_B + dy) → (x_B, y_B).
Les deux points sont bien à l'intérieur de Frame B pour les têtes « stayed ».

Structure de sortie générée sous ``data_result/`` :
    {train, val, test}/
        viz/     ← images annotées (.jpg)
        labels/  ← fichiers de labels numériques (.txt)

Lit depuis  : data_tracking/{split}/
Écrit dans  : data_result/{split}/

Dépendances Python :
    opencv-python, numpy
"""

import os

import cv2
import numpy as np


# =============================================================================
# Lecture des fichiers du dataset de tracking
# =============================================================================

def _load_gt_file(path: str) -> list[dict]:
    """
    Lit un fichier d'annotations (gt_A/*.txt ou gt_B/*.txt).

    Format attendu (une ligne par tête) :
        x  y  w  h  occlusion  blur

    Args:
        path (str) : chemin absolu vers le fichier .txt.

    Returns:
        list[dict] : liste des têtes annotées (vide si fichier absent).
    """
    anns = []
    if not os.path.exists(path):
        return anns
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                anns.append({
                    "x": int(parts[0]),
                    "y": int(parts[1]),
                    "w": int(parts[2]),
                    "h": int(parts[3]),
                    "occlusion": int(parts[4]),
                    "blur":      int(parts[5]),
                })
    return anns


def _load_links_file(path: str) -> list[tuple[int, int]]:
    """
    Lit un fichier de liens de correspondance (gt_links/*.txt).

    Format (une ligne par lien « stayed ») :
        idx_A  idx_B

    Args:
        path (str) : chemin absolu vers le fichier .txt.

    Returns:
        list[tuple[int,int]] : paires (idx_dans_gt_A, idx_dans_gt_B).
    """
    links = []
    if not os.path.exists(path):
        return links
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                links.append((int(parts[0]), int(parts[1])))
    return links


def _load_image_labels(labels_file: str) -> dict:
    """
    Lit le fichier image_labels.txt du dataset de tracking.

    Format (une ligne par paire) :
        id,count_A,count_B,count_stayed,count_left,count_entered,scene,weather,distractor

    Args:
        labels_file (str) : chemin absolu vers le fichier.

    Returns:
        dict[str, dict] : clés = identifiants d'images, valeurs = métadonnées.
    """
    labels = {}
    with open(labels_file, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 9:
                continue
            labels[parts[0]] = {
                "count_A":       int(parts[1]),
                "count_B":       int(parts[2]),
                "count_stayed":  int(parts[3]),
                "count_left":    int(parts[4]),
                "count_entered": int(parts[5]),
                "scene":         parts[6],
                "weather":       int(parts[7]),
                "distractor":    int(parts[8]),
            }
    return labels


# =============================================================================
# Inférence du décalage global (dx, dy) à partir des liens
# =============================================================================

def _infer_global_shift(
    gt_A: list[dict],
    gt_B: list[dict],
    links: list[tuple[int, int]],
) -> tuple[int, int]:
    """
    Déduit le décalage global (dx, dy) entre Frame A et Frame B.

    Principe :
        Pour toute tête « stayed », on sait que :
            x_B = x_A - dx   →   dx = x_A - x_B
            y_B = y_A - dy   →   dy = y_A - y_B

        On utilise le PREMIER lien disponible pour calculer dx et dy.
        Toutes les paires devraient donner le même résultat puisque le décalage
        est GLOBAL (il est le même pour toutes les têtes).

    Args:
        gt_A (list[dict])              : annotations de Frame A.
        gt_B (list[dict])              : annotations de Frame B.
        links (list[tuple[int,int]])   : paires (idx_A, idx_B) des têtes « stayed ».

    Returns:
        tuple[int, int] : (dx, dy) en pixels. Retourne (0, 0) si aucun lien.
    """
    if not links:
        # Sans lien, on ne peut pas déduire le décalage
        # Les visualisations n'auront pas de flèches, seulement des cercles.
        return 0, 0

    idx_A, idx_B = links[0]
    dx = gt_A[idx_A]["x"] - gt_B[idx_B]["x"]
    dy = gt_A[idx_A]["y"] - gt_B[idx_B]["y"]
    return dx, dy


# =============================================================================
# Génération de l'image de visualisation
# =============================================================================

def generate_visualization(
    frame_B_bgr: np.ndarray,
    gt_A: list[dict],
    gt_B: list[dict],
    links: list[tuple[int, int]],
    line_thickness: int = 1,
    marker_radius: int = 4,
    draw_legend: bool = True,
) -> np.ndarray:
    """
    Génère une image de visualisation couleur (BGR) annotée avec des marqueurs noirs.

    L'image de base est Frame B conservée en couleur.  Trois types de
    marqueurs noirs sont dessinés dessus, un par type d'événement de tracking :

    1. TRAIT NOIR simple — têtes « stayed » (présentes dans Frame A et Frame B)
       Un segment relie la position ancienne (Frame A, exprimée en coords Frame B)
       à la position actuelle (Frame B).  Pas de pointe de flèche : seule la
       correspondance spatiale est montrée, pas la direction.
       Formule du point de départ : (x_prev, y_prev) = (x_B + dx, y_B + dy)
       Ces deux coordonnées sont garanties à l'intérieur de l'image pour les
       têtes « stayed ».

    2. POINT NOIR plein — têtes « entered » (+1)
       Petit disque plein à la position (x_B, y_B) de la tête dans Frame B.
       Signifie : cette personne est apparue dans le champ, elle n'était pas
       visible dans Frame A.

    3. CROIX NOIRE (×) — têtes « left » (-1)
       Deux segments en X centrés sur la position de sortie, clampée à la
       bordure de Frame B.  Signifie : cette personne a quitté le champ entre
       Frame A et Frame B.

    Args:
        frame_B_bgr (np.ndarray) :
            Image Frame B au format BGR (OpenCV), dtype uint8, shape (H, W, 3).
        gt_A (list[dict]) : annotations des têtes dans Frame A.
        gt_B (list[dict]) : annotations des têtes dans Frame B.
        links (list[tuple[int,int]]) :
            Paires (idx_A, idx_B) des têtes appariées (stayed).
        line_thickness (int) :
            Épaisseur des traits et des bras de croix en pixels.  Défaut 1.
        marker_radius (int) :
            Rayon du point plein (entered) et demi-largeur de la croix (left).
            Défaut 4 pixels.
        draw_legend (bool) :
            Si True, ajoute une légende en bas à gauche de l'image.

    Returns:
        np.ndarray : image BGR couleur annotée, shape (H, W, 3), dtype uint8.
    """
    # Copie pour ne pas modifier l'image originale
    vis = frame_B_bgr.copy()
    H, W = vis.shape[:2]

    # Couleur unique pour tous les marqueurs : noir pur en BGR
    NOIR = (0, 0, 0)

    # Inférence du décalage global à partir des liens (dx, dy en pixels)
    dx, dy = _infer_global_shift(gt_A, gt_B, links)

    # Ensembles d'indices pour distinguer stayed / entered / left
    stayed_A_idx: set[int] = {idx_A for idx_A, _     in links}
    stayed_B_idx: set[int] = {idx_B for _,     idx_B in links}

    # -------------------------------------------------------------------------
    # Stayed : trait noir simple (sans pointe de flèche, sans direction)
    # -------------------------------------------------------------------------
    for idx_A, idx_B in links:
        x_B, y_B = gt_B[idx_B]["x"], gt_B[idx_B]["y"]

        # Position de la tête dans Frame A, exprimée en coords pixel de Frame B.
        # Relation : x_A = x_B + dx,  y_A = y_B + dy.
        # Pour les têtes stayed, ces deux valeurs sont garanties dans [0,W)×[0,H).
        x_prev = x_B + dx
        y_prev = y_B + dy

        # Segment noir reliant l'ancienne position à la nouvelle
        cv2.line(vis, (x_prev, y_prev), (x_B, y_B), NOIR, thickness=line_thickness)

    # -------------------------------------------------------------------------
    # Entered : point noir plein (+1)
    # -------------------------------------------------------------------------
    # thickness=-1 remplit entièrement le cercle (disque plein)
    for idx_B, a_B in enumerate(gt_B):
        if idx_B in stayed_B_idx:
            continue  # already drawn as stayed
        cv2.circle(vis, (a_B["x"], a_B["y"]),
                   radius=marker_radius, color=NOIR, thickness=-1)

    # -------------------------------------------------------------------------
    # Left : croix noire × (-1)
    # -------------------------------------------------------------------------
    # La croix est formée de deux segments diagonaux qui se croisent au centre.
    # Elle est dessinée à la position de sortie (clampée à la bordure de Frame B).
    for idx_A, a_A in enumerate(gt_A):
        if idx_A in stayed_A_idx:
            continue  # already drawn as stayed

        # Position de sortie : (x_A - dx, y_A - dy) peut être négative.
        # np.clip la ramène dans [0, dimension - 1].
        x_exit = int(np.clip(a_A["x"] - dx, 0, W - 1))
        y_exit = int(np.clip(a_A["y"] - dy, 0, H - 1))

        r = marker_radius  # demi-largeur de la croix en pixels
        # Bras diagonal ↘
        cv2.line(vis, (x_exit - r, y_exit - r), (x_exit + r, y_exit + r),
                 NOIR, thickness=line_thickness)
        # Bras diagonal ↗
        cv2.line(vis, (x_exit + r, y_exit - r), (x_exit - r, y_exit + r),
                 NOIR, thickness=line_thickness)

    # -------------------------------------------------------------------------
    # Légende (coin inférieur gauche)
    # -------------------------------------------------------------------------
    if draw_legend:
        n_stayed  = len(links)
        n_entered = len(gt_B) - len(stayed_B_idx)
        n_left    = len(gt_A) - len(stayed_A_idx)

        # Rectangle blanc (255) comme fond de légende pour garantir la lisibilité
        # quelle que soit la luminosité du fond de l'image
        legend_h, legend_w = 72, 188
        x_leg, y_leg = 8, H - legend_h - 8
        cv2.rectangle(vis,
                      (x_leg, y_leg),
                      (x_leg + legend_w, y_leg + legend_h),
                      (255, 255, 255), thickness=-1)

        # Trois lignes : symbole + description + comptage
        entries = [
            ("ligne",  f"stayed  : {n_stayed}"),
            ("point",  f"entered : {n_entered}  (+1)"),
            ("croix",  f"left    : {n_left}  (-1)"),
        ]
        for k, (symbole, texte) in enumerate(entries):
            y_item = y_leg + 18 + k * 22
            cx_sym = x_leg + 13   # centre horizontal du symbole dans la légende

            # Dessin du symbole correspondant au type d'événement
            if symbole == "ligne":
                # Petit segment horizontal représentant le trait stayed
                cv2.line(vis, (x_leg + 5, y_item), (x_leg + 21, y_item), NOIR, 1)

            elif symbole == "point":
                # Disque plein représentant le point entered
                cv2.circle(vis, (cx_sym, y_item - 1),
                           radius=marker_radius - 1, color=NOIR, thickness=-1)

            else:  # "croix"
                # Croix × représentant le marqueur left
                r = marker_radius - 1
                cv2.line(vis, (cx_sym - r, y_item - 1 - r),
                               (cx_sym + r, y_item - 1 + r), NOIR, 1)
                cv2.line(vis, (cx_sym + r, y_item - 1 - r),
                               (cx_sym - r, y_item - 1 + r), NOIR, 1)

            # Texte descriptif à droite du symbole
            cv2.putText(vis, texte,
                        (x_leg + 28, y_item + 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.38,
                        color=NOIR,
                        thickness=1)

    return vis


# =============================================================================
# Génération des labels numériques
# =============================================================================

def generate_label_data(
    gt_A: list[dict],
    gt_B: list[dict],
    links: list[tuple[int, int]],
    img_h: int,
    img_w: int,
) -> list[dict]:
    """
    Génère la liste des événements de tracking pour une paire de frames.

    Chaque événement décrit le sort d'une tête entre Frame A et Frame B :
        • status  0  = stayed  : la tête est présente dans les deux frames.
        • status +1  = entered : la tête est apparue dans Frame B (inconnue de A).
        • status -1  = left    : la tête était dans Frame A, absente de Frame B.

    Les coordonnées sont toutes exprimées dans l'espace pixel de Frame B.
    Pour les têtes « left », dont la position en Frame B serait hors-image
    (valeur négative), la position est clampée à la bordure la plus proche.

    Les déplacements (delta_x, delta_y) représentent le vecteur de déplacement
    de Frame A vers Frame B dans l'espace pixel de Frame B :
        stayed  : (delta_x, delta_y) = (x_B - x_A, y_B - y_A) = (-dx_global, -dy_global)
        entered : (delta_x, delta_y) = (0, 0) — pas de position en Frame A connue
        left    : (delta_x, delta_y) = (x_exit - x_A, y_exit - y_A)
                   où x_exit, y_exit sont les coordonnées clampées

    Args:
        gt_A (list[dict])            : annotations des têtes dans Frame A.
        gt_B (list[dict])            : annotations des têtes dans Frame B.
        links (list[tuple[int,int]]) : paires (idx_A, idx_B) des têtes « stayed ».
        img_h (int) : hauteur de Frame B en pixels (pour le clamp des têtes « left »).
        img_w (int) : largeur de Frame B en pixels.

    Returns:
        list[dict] : liste d'événements de tracking.  Chaque dict contient :
            - "status"  (int)   :  0=stayed  +1=entered  -1=left
            - "x_B"     (int)   : position x dans Frame B (ou position clampée)
            - "y_B"     (int)   : position y dans Frame B (ou position clampée)
            - "delta_x" (int)   : déplacement horizontal Frame A → Frame B
            - "delta_y" (int)   : déplacement vertical  Frame A → Frame B
    """
    # Déduction du décalage global depuis les liens
    dx, dy = _infer_global_shift(gt_A, gt_B, links)

    # Index des têtes « stayed » dans gt_A et gt_B (pour identifier entered/left)
    stayed_A_idx: set[int] = {idx_A for idx_A, _     in links}
    stayed_B_idx: set[int] = {idx_B for _,     idx_B in links}

    events: list[dict] = []

    # ---- Événements « stayed » -----------------------------------------------
    for idx_A, idx_B in links:
        x_B = gt_B[idx_B]["x"]
        y_B = gt_B[idx_B]["y"]
        x_A = gt_A[idx_A]["x"]  # = x_B + dx
        y_A = gt_A[idx_A]["y"]  # = y_B + dy

        events.append({
            "status":  0,
            "x_B":     x_B,
            "y_B":     y_B,
            "delta_x": x_B - x_A,   # = -dx_global (déplacement négatif : vers la gauche/haut)
            "delta_y": y_B - y_A,   # = -dy_global
        })

    # ---- Événements « entered » (+1) -----------------------------------------
    # Têtes dans gt_B sans lien vers gt_A → nouvelles dans Frame B
    for idx_B, a_B in enumerate(gt_B):
        if idx_B in stayed_B_idx:
            continue  # déjà comptée dans stayed

        events.append({
            "status":  1,     # +1 : entrée dans le champ
            "x_B":     a_B["x"],
            "y_B":     a_B["y"],
            "delta_x": 0,     # pas d'historique Frame A → déplacement inconnu
            "delta_y": 0,
        })

    # ---- Événements « left » (-1) --------------------------------------------
    # Têtes dans gt_A sans lien vers gt_B → parties entre Frame A et Frame B
    for idx_A, a_A in enumerate(gt_A):
        if idx_A in stayed_A_idx:
            continue  # déjà comptée dans stayed

        x_A, y_A = a_A["x"], a_A["y"]

        # Position de sortie dans l'espace Frame B :
        # x_A - dx est négatif pour une tête sortie par le bord gauche ou haut.
        # On clamp dans [0, img_w-1] × [0, img_h-1] pour rester dans l'image.
        x_exit = int(np.clip(x_A - dx, 0, img_w - 1))
        y_exit = int(np.clip(y_A - dy, 0, img_h - 1))

        events.append({
            "status":  -1,             # -1 : sortie du champ
            "x_B":     x_exit,         # position de sortie, clampée à la bordure
            "y_B":     y_exit,
            "delta_x": x_exit - x_A,   # vecteur vers le point de sortie
            "delta_y": y_exit - y_A,
        })

    return events


# =============================================================================
# Sauvegarde du fichier de labels
# =============================================================================

def save_label_file(path: str, label_data: list[dict]) -> None:
    """
    Sauvegarde les labels numériques dans un fichier texte lisible.

    Format de sortie (une ligne par événement) :
        status  x_B  y_B  delta_x  delta_y

    Avec en en-tête deux lignes de commentaires qui décrivent le format
    pour qu'un lecteur humain puisse comprendre le fichier sans documentation.

    Exemple de contenu :
        # status x_B y_B delta_x delta_y
        # status : 0=stayed  +1=entered  -1=left
        0 245 130 -28 -22
        0 310  90 -28 -22
        1 450 280   0   0
       -1   0  15 -32 -18

    Args:
        path (str)            : chemin absolu du fichier à créer.
        label_data (list[dict]) : liste d'événements retournée par generate_label_data.

    Returns:
        None
    """
    # Création des dossiers parents si nécessaire
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        # En-tête descriptif — permet de comprendre le fichier sans documentation externe
        f.write("# status x_B y_B delta_x delta_y\n")
        f.write("# status : 0=stayed  +1=entered  -1=left\n")

        for ev in label_data:
            # Format aligné pour la lisibilité humaine
            f.write(
                f"{ev['status']:>2}  "
                f"{ev['x_B']:>5}  {ev['y_B']:>5}  "
                f"{ev['delta_x']:>7}  {ev['delta_y']:>7}\n"
            )


# =============================================================================
# Traitement d'une paire d'images
# =============================================================================

def _process_pair(
    img_id: str,
    tracking_split_dir: str,
    output_split_dir: str,
) -> bool:
    """
    Traite une paire (Frame A, Frame B) et génère la visualisation et les labels.

    Pipeline complet pour une image :
        1. Chargement de Frame B depuis frames_B/
        2. Chargement des annotations gt_A, gt_B et des liens gt_links
        3. Génération de l'image de visualisation annotée
        4. Génération des labels numériques
        5. Sauvegarde de la visualisation dans viz/ et des labels dans labels/

    Args:
        img_id (str)              : identifiant de l'image (ex : "0001").
        tracking_split_dir (str)  : chemin vers le split dans data_tracking/.
        output_split_dir (str)    : chemin vers le split de sortie dans data_result/.

    Returns:
        bool : True si le traitement a réussi, False si Frame B est illisible.
    """
    # --- Chargement de Frame B ---
    # C'est l'image de fond sur laquelle on dessine les annotations
    frame_B_path = os.path.join(tracking_split_dir, "frames_B", f"{img_id}.jpg")
    frame_B = cv2.imread(frame_B_path)
    if frame_B is None:
        print(f"  [AVERTISSEMENT] Frame B illisible : {frame_B_path}")
        return False

    img_h, img_w = frame_B.shape[:2]

    # --- Chargement des annotations ---
    gt_A   = _load_gt_file(   os.path.join(tracking_split_dir, "gt_A",     f"{img_id}.txt"))
    gt_B   = _load_gt_file(   os.path.join(tracking_split_dir, "gt_B",     f"{img_id}.txt"))
    links  = _load_links_file(os.path.join(tracking_split_dir, "gt_links", f"{img_id}.txt"))

    # --- Génération de la visualisation ---
    viz_image = generate_visualization(
        frame_B_bgr    = frame_B,
        gt_A           = gt_A,
        gt_B           = gt_B,
        links          = links,
        line_thickness = 1,
        marker_radius  = 6,
        draw_legend    = True,
    )

    # --- Génération des labels numériques ---
    label_data = generate_label_data(
        gt_A  = gt_A,
        gt_B  = gt_B,
        links = links,
        img_h = img_h,
        img_w = img_w,
    )

    # --- Sauvegarde de la visualisation ---
    viz_path = os.path.join(output_split_dir, "viz", f"{img_id}.jpg")
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    cv2.imwrite(viz_path, viz_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # --- Sauvegarde du fichier de labels ---
    label_path = os.path.join(output_split_dir, "labels", f"{img_id}.txt")
    save_label_file(label_path, label_data)

    return True


# =============================================================================
# Fonction principale — traitement de tous les splits
# =============================================================================

def build_result_dataset(
    tracking_root: str,
    output_root: str,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """
    Génère le dataset de résultats complet à partir du dataset de tracking.

    Pour chaque split ("train", "val", "test"), cette fonction :
        1. Lit les métadonnées (image_labels.txt) depuis data_tracking/
        2. Pour chaque paire, appelle ``_process_pair`` qui génère :
               - l'image de visualisation annotée (viz/)
               - le fichier de labels numériques  (labels/)
        3. Affiche des statistiques résumées (total stayed/entered/left)

    Prérequis :
        Le dataset data_tracking/ doit avoir été généré au préalable
        par ``dataCleaning.build_tracking_dataset``.

    Args:
        tracking_root (str) : chemin vers le dossier ``data_tracking/``.
        output_root (str)   : chemin vers le dossier de sortie ``data_result/``.
        splits (tuple)      : liste des splits à traiter.

    Returns:
        None
    """
    for split in splits:
        print(f"\n{'='*60}")
        print(f"  Split : {split.upper()}")
        print(f"{'='*60}")

        tracking_split_dir = os.path.join(tracking_root, split)
        output_split_dir   = os.path.join(output_root,   split)

        # Vérification que le split de tracking existe
        labels_file = os.path.join(tracking_split_dir, "image_labels.txt")
        if not os.path.exists(labels_file):
            print(f"  [ERREUR] Fichier introuvable : {labels_file}")
            print(f"           → Lancez d'abord dataCleaning.build_tracking_dataset()")
            continue

        # Chargement des métadonnées pour obtenir la liste des identifiants
        image_labels = _load_image_labels(labels_file)
        img_ids      = list(image_labels.keys())
        print(f"  {len(img_ids)} paires à traiter...")

        ok_count = 0
        errors   = 0

        # Accumulateurs pour les statistiques globales du split
        total_stayed  = 0
        total_entered = 0
        total_left    = 0

        for i, img_id in enumerate(img_ids):
            # Progression toutes les 200 images
            if (i + 1) % 200 == 0 or (i + 1) == len(img_ids):
                print(f"  [{i + 1:>4}/{len(img_ids)}] ...")

            success = _process_pair(img_id, tracking_split_dir, output_split_dir)

            if success:
                ok_count += 1
                # Accumulation des statistiques depuis les métadonnées
                meta = image_labels[img_id]
                total_stayed  += meta["count_stayed"]
                total_entered += meta["count_entered"]
                total_left    += meta["count_left"]
            else:
                errors += 1

        # Affichage des statistiques du split
        print(f"\n  Résultats {split.upper()} :")
        print(f"    Paires traitées : {ok_count:>5} / {len(img_ids)}")
        print(f"    Erreurs         : {errors:>5}")
        print(f"    Stayed total    : {total_stayed:>8,}")
        print(f"    Entered total   : {total_entered:>8,}")
        print(f"    Left total      : {total_left:>8,}")
        print(f"  Visualisations → {os.path.join(output_split_dir, 'viz')}")
        print(f"  Labels         → {os.path.join(output_split_dir, 'labels')}")

    print(f"\n\nDataset de résultats généré dans : {output_root}")


# =============================================================================
# Test rapide sur une seule image
# =============================================================================

def smoke_test(tracking_root: str, split: str = "train") -> None:
    """
    Teste le pipeline sur la première image disponible du split.

    Génère la visualisation et les labels pour une seule image et les affiche
    dans la console pour vérifier que tout fonctionne avant le traitement complet.

    Args:
        tracking_root (str) : chemin vers data_tracking/.
        split (str)         : split à tester ("train", "val" ou "test").
    """
    labels_file = os.path.join(tracking_root, split, "image_labels.txt")
    if not os.path.exists(labels_file):
        print(f"[ERREUR] {labels_file} introuvable.")
        return

    image_labels = _load_image_labels(labels_file)
    if not image_labels:
        print("[ERREUR] Aucune image dans le fichier labels.")
        return

    img_id = next(iter(image_labels))   # premier identifiant
    split_dir = os.path.join(tracking_root, split)

    frame_B_path = os.path.join(split_dir, "frames_B", f"{img_id}.jpg")
    frame_B      = cv2.imread(frame_B_path)

    gt_A  = _load_gt_file(   os.path.join(split_dir, "gt_A",     f"{img_id}.txt"))
    gt_B  = _load_gt_file(   os.path.join(split_dir, "gt_B",     f"{img_id}.txt"))
    links = _load_links_file(os.path.join(split_dir, "gt_links",  f"{img_id}.txt"))

    print(f"\n--- Smoke test : image {img_id} ({split}) ---")
    print(f"  Frame B shape : {frame_B.shape if frame_B is not None else 'ERREUR'}")
    print(f"  Têtes gt_A    : {len(gt_A)}")
    print(f"  Têtes gt_B    : {len(gt_B)}")
    print(f"  Liens         : {len(links)}")

    dx, dy = _infer_global_shift(gt_A, gt_B, links)
    print(f"  Shift inféré  : dx={dx}  dy={dy}")

    if frame_B is not None:
        img_h, img_w = frame_B.shape[:2]
        labels = generate_label_data(gt_A, gt_B, links, img_h, img_w)
        stayed  = [e for e in labels if e["status"] ==  0]
        entered = [e for e in labels if e["status"] ==  1]
        left    = [e for e in labels if e["status"] == -1]

        print(f"  Labels générés : {len(labels)} événements")
        print(f"    stayed  : {len(stayed)}")
        print(f"    entered : {len(entered)}")
        print(f"    left    : {len(left)}")

        if stayed:
            s = stayed[0]
            print(f"  Exemple stayed  → x_B={s['x_B']} y_B={s['y_B']} "
                  f"delta=({s['delta_x']}, {s['delta_y']})")
        if entered:
            e = entered[0]
            print(f"  Exemple entered → x_B={e['x_B']} y_B={e['y_B']}")
        if left:
            l = left[0]
            print(f"  Exemple left    → x_exit={l['x_B']} y_exit={l['y_B']} "
                  f"delta=({l['delta_x']}, {l['delta_y']})")

        print("\n  Smoke test OK")
    else:
        print("  [ERREUR] Impossible de lire Frame B — smoke test interrompu.")


# =============================================================================
# Point d'entrée — lancement direct du script
# =============================================================================

if __name__ == "__main__":
    TRACKING_ROOT = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data_tracking")
    )
    OUTPUT_ROOT = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data_result")
    )

    print(f"Source      : {TRACKING_ROOT}")
    print(f"Destination : {OUTPUT_ROOT}")

    # Vérification préalable que le dataset de tracking existe
    if not os.path.isdir(TRACKING_ROOT):
        print("\n[ERREUR] Le dossier data_tracking/ n'existe pas.")
        print("  → Lancez d'abord : python dataCleaning.py")
    else:
        # Test rapide sur une image avant le traitement complet
        smoke_test(TRACKING_ROOT, split="train")

        # Traitement complet des trois splits
        build_result_dataset(
            tracking_root = TRACKING_ROOT,
            output_root   = OUTPUT_ROOT,
            splits        = ("train", "val", "test"),
        )
