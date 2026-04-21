"""
dataCleaning.py — Génération du dataset de tracking de foule
=============================================================

Ce script transforme le dataset de comptage statique JHU-CROWD++ v2.0
en un dataset de tracking entre paires de frames vidéo simulées.

Principe de simulation
----------------------
Un vrai système de tracking vidéo analyse deux frames successives et
détermine quelles personnes sont passées de l'une à l'autre.  Comme notre
dataset d'origine ne contient que des images statiques (pas de vidéos), on
simule deux frames à partir d'une seule image en utilisant un **décalage de
champ de caméra** (crop shift) :

    Frame A = crop de l'image originale depuis le coin supérieur-gauche
              Région : [0 : H - dy,  0 : W - dx]

    Frame B = crop de la même image depuis un coin décalé
              Région : [dy : H,  dx : W]

                     image originale (W × H)
    ┌─────────────────────────────────────────┐
    │╔══════════════════════════╗             │
    │║      Frame A             ║  ← crop     │
    │║    (W-dx) × (H-dy)       ║             │
    │║                    ╔════════════════╗  │
    │╚════════════════════╬════╝           │  │
    │         ↑ zone de   ║    Frame B     │  │
    │         transition  ║ (W-dx)×(H-dy)  │  │
    │                     ╚════════════════╝  │
    └─────────────────────────────────────────┘

Grâce à ce décalage, chaque tête annotée se retrouve dans l'une des trois
situations suivantes :

  • stayed  : visible dans Frame A ET Frame B (zone de chevauchement)
              → sa position diffère de (dx, dy) entre les deux frames
  • left    : visible uniquement dans Frame A (bord gauche/haut)
              → la personne a "quitté le champ" entre les deux frames
  • entered : visible uniquement dans Frame B (bord droit/bas)
              → la personne est "entrée dans le champ" entre les deux frames

C'est exactement l'information nécessaire pour entraîner un tracker :
le modèle doit apprendre à faire des liens entre têtes d'une frame à l'autre.

Fichiers générés (sous data_tracking/)
---------------------------------------
data_tracking/
    {train, val, test}/
        frames_A/          ← images .jpg (crop Frame A)
        frames_B/          ← images .jpg (crop Frame B, même taille que A)
        gt_A/              ← annotations des têtes dans Frame A (même format que gt/)
        gt_B/              ← annotations des têtes dans Frame B
        gt_links/          ← correspondances tête-à-tête entre Frame A et Frame B
        image_labels.txt   ← métadonnées tracking par paire d'images

Format gt_A/*.txt et gt_B/*.txt (identique au format original) :
    x  y  w  h  occlusion  blur
    (une ligne = une tête)

Format gt_links/*.txt :
    idx_A  idx_B
    (une ligne = une tête appariée ; idx = numéro de ligne dans gt_A / gt_B)
    Têtes dans gt_A SANS lien correspondant → "left" (parties)
    Têtes dans gt_B SANS lien correspondant → "entered" (arrivées)

Format image_labels.txt (tracking) :
    id, count_A, count_B, count_stayed, count_left, count_entered, scene, weather, distractor

Paramètres clés
---------------
  shift_range (int) : décalage maximum en pixels entre les deux crops.
      Valeur par défaut : 30 px.  Représente une caméra qui "pan" légèrement.
      Automatiquement plafonné à 5 % de la plus petite dimension de l'image.

  brightness_jitter (float) : amplitude maximale de variation de luminosité globale (±).
      Par défaut 0.08 → la frame peut être jusqu'à 8 % plus claire ou plus sombre.

  color_jitter (float) : amplitude maximale du décalage colorimétrique par canal (±).
      Par défaut 0.05 → chaque canal BGR peut varier indépendamment de ±5 %,
      simulant une légère dominante colorée (blanc chaud, froid, verdâtre…).

  noise_sigma (float) : écart-type du bruit Gaussien additif (sur l'échelle 0-255).
      Par défaut 3.0 → bruit très fin, à peine perceptible à l'œil mais présent
      dans toute image issue d'un capteur CCD/CMOS.

  seed (int) : graine du générateur aléatoire pour la reproductibilité.
      Même graine → même dataset généré.

Dépendances Python
------------------
    opencv-python, numpy
"""

import os

import cv2
import numpy as np


# =============================================================================
# Parsing des fichiers sources (lecture du dataset original)
# =============================================================================

def _parse_image_labels(labels_file: str) -> dict:
    """
    Lit et parse le fichier ``image_labels.txt`` du dataset original.

    Chaque ligne a le format :
        id,count,scene,weather,distractor
    Exemple :
        0001,161,water park,0,0

    Args:
        labels_file (str): chemin absolu vers le fichier image_labels.txt.

    Returns:
        dict[str, dict] : clés = identifiants d'images (ex : "0001"),
        valeurs = dicts avec les champs :
            - "count"      (int)  : nombre de personnes
            - "scene"      (str)  : type de lieu ("stadium", "street"…)
            - "weather"    (int)  : code météo  0=clair 1=brouillard 2=pluie 3=neige
            - "distractor" (int)  : 1 si image distracteur, 0 sinon
    """
    labels = {}

    with open(labels_file, "r") as f:
        for line in f:
            # Suppression des espaces parasites, découpage sur les virgules
            parts = [p.strip() for p in line.strip().split(",")]

            # Ignore les lignes incomplètes ou vides
            if len(parts) < 5:
                continue

            img_id = parts[0]
            labels[img_id] = {
                "count": int(parts[1]),
                "scene": parts[2],
                "weather": int(parts[3]),
                "distractor": int(parts[4]),
            }

    return labels


def _parse_gt_file(gt_path: str) -> list[dict]:
    """
    Lit un fichier d'annotation tête par tête (``gt/*.txt``).

    Chaque ligne a le format :
        x  y  w  h  occlusion  blur
    Exemple :
        106 114 24 25 1 0

    Args:
        gt_path (str): chemin absolu vers le fichier .txt d'annotations.

    Returns:
        list[dict] : liste des têtes annotées.  Chaque dict contient :
            - "x"         (int) : colonne du centre de la tête (pixels)
            - "y"         (int) : ligne du centre de la tête (pixels)
            - "w"         (int) : largeur approximative (pixels)
            - "h"         (int) : hauteur approximative (pixels)
            - "occlusion" (int) : 1=visible 2=partielle 3=totale
            - "blur"      (int) : 0=nette 1=floue
        Retourne une liste vide si le fichier n'existe pas.
    """
    anns = []

    # Certaines images n'ont pas de fichier gt (0 personne dans la scène)
    if not os.path.exists(gt_path):
        return anns

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                anns.append({
                    "x": int(parts[0]),
                    "y": int(parts[1]),
                    "w": int(parts[2]),
                    "h": int(parts[3]),
                    "occlusion": int(parts[4]),
                    "blur": int(parts[5]),
                })

    return anns


# =============================================================================
# Calcul du décalage (shift)
# =============================================================================

def _compute_shift(
    orig_w: int,
    orig_h: int,
    shift_range: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """
    Calcule un décalage aléatoire (dx, dy) pour séparer Frame A de Frame B.

    Le décalage est tiré uniformément dans l'intervalle [min_shift, max_shift]
    pour chaque axe indépendamment.  Le maximum est plafonné à 5 % de la
    dimension de l'image pour éviter de perdre trop de têtes.

    Exemple pour une image 1000 × 800 avec shift_range=30 :
        max_dx = min(30, 50) = 30   → dx ∈ [5, 30]
        max_dy = min(30, 40) = 30   → dy ∈ [5, 30]

    Args:
        orig_w (int)  : largeur de l'image originale en pixels.
        orig_h (int)  : hauteur de l'image originale en pixels.
        shift_range (int) : décalage maximum souhaité en pixels.
        rng (np.random.Generator) : générateur de nombres aléatoires.

    Returns:
        tuple[int, int] : (dx, dy) — décalage horizontal et vertical en pixels.
    """
    # Limite supérieure : ne pas dépasser 5 % de la dimension concernée
    max_dx = min(shift_range, max(5, int(0.05 * orig_w)))
    max_dy = min(shift_range, max(5, int(0.05 * orig_h)))

    # Limite inférieure : au moins 5 pixels pour avoir un vrai décalage
    # (ou moins si l'image est vraiment très petite)
    min_dx = min(5, max_dx)
    min_dy = min(5, max_dy)

    # Tirage uniforme entier dans [min, max]
    dx = int(rng.integers(min_dx, max_dx + 1))
    dy = int(rng.integers(min_dy, max_dy + 1))

    return dx, dy


# =============================================================================
# Génération des annotations de la paire (Frame A, Frame B)
# =============================================================================

def _generate_pair(
    anns: list[dict],
    dx: int,
    dy: int,
    orig_w: int,
    orig_h: int,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """
    Répartit les annotations originales entre Frame A, Frame B et les liens.

    Logique de répartition
    ~~~~~~~~~~~~~~~~~~~~~~
    Les deux frames sont des crops de l'image originale :

        Frame A = image[0 : H-dy,  0 : W-dx]   (coin supérieur-gauche)
        Frame B = image[dy : H,    dx : W]      (coin inférieur-droit)

    Pour chaque tête annotée à la position (xi, yi) dans l'image originale :

        dans Frame A  si  xi < W-dx  ET  yi < H-dy
        dans Frame B  si  xi ≥ dx    ET  yi ≥ dy

        dans les DEUX (stayed) si  dx ≤ xi < W-dx  ET  dy ≤ yi < H-dy
        dans A seulement (left)    si dans A mais pas dans B
        dans B seulement (entered) si dans B mais pas dans A

    Les coordonnées dans Frame B sont décalées :
        x_B = xi - dx
        y_B = yi - dy
    car le crop B commence à (dx, dy) dans l'image originale.

    Args:
        anns (list[dict]) : toutes les têtes de l'image originale.
        dx (int)   : décalage horizontal en pixels.
        dy (int)   : décalage vertical en pixels.
        orig_w (int) : largeur de l'image originale.
        orig_h (int) : hauteur de l'image originale.

    Returns:
        tuple de trois éléments :
            gt_A (list[dict])              : annotations dans Frame A
                                             (coordonnées originales inchangées)
            gt_B (list[dict])              : annotations dans Frame B
                                             (coordonnées relatives au crop B)
            links (list[tuple[int,int]])   : paires (idx_dans_gt_A, idx_dans_gt_B)
                                             pour les têtes "stayed"
    """
    # Dimensions des crops (Frame A et Frame B ont la même taille)
    crop_w = orig_w - dx
    crop_h = orig_h - dy

    gt_A = []    # têtes valides dans Frame A
    gt_B = []    # têtes valides dans Frame B (coordonnées relatives au crop B)

    # Dictionnaires de correspondance :
    # orig_to_A[i] = indice de la tête i dans gt_A  (si elle est dans Frame A)
    # orig_to_B[i] = indice de la tête i dans gt_B  (si elle est dans Frame B)
    orig_to_A: dict[int, int] = {}
    orig_to_B: dict[int, int] = {}

    for i, a in enumerate(anns):
        xi, yi = a["x"], a["y"]

        # --- Appartenance à Frame A ---
        # Frame A = crop [0:crop_h, 0:crop_w]
        # Le centre de la tête doit être strictement à l'intérieur du crop
        in_A = (0 <= xi < crop_w) and (0 <= yi < crop_h)

        # --- Appartenance à Frame B ---
        # Frame B = crop [dy:orig_h, dx:orig_w]
        # En coordonnées originales, la tête doit être dans [dx, orig_w) × [dy, orig_h)
        # Note : xi < orig_w et yi < orig_h sont toujours vrais pour des annotations valides
        in_B = (xi >= dx) and (yi >= dy)

        if in_A:
            # Coordonnées dans Frame A = coordonnées originales (pas de décalage)
            orig_to_A[i] = len(gt_A)
            gt_A.append(dict(a))

        if in_B:
            # Coordonnées dans Frame B = originales décalées de (-dx, -dy)
            # car le crop commence en (dx, dy) dans l'image originale
            ann_b = dict(a)          # copie du dictionnaire original
            ann_b["x"] = xi - dx    # position relative dans Frame B
            ann_b["y"] = yi - dy
            orig_to_B[i] = len(gt_B)
            gt_B.append(ann_b)

    # --- Construction des liens (stayed = présentes dans les deux frames) ---
    links: list[tuple[int, int]] = []
    for i in range(len(anns)):
        # Une tête est "stayed" si et seulement si elle apparaît dans A ET dans B
        if i in orig_to_A and i in orig_to_B:
            links.append((orig_to_A[i], orig_to_B[i]))

    return gt_A, gt_B, links


# =============================================================================
# Sauvegarde des fichiers générés
# =============================================================================

def _save_gt_file(path: str, anns: list[dict]) -> None:
    """
    Sauvegarde une liste d'annotations au format gt/*.txt standard.

    Format de sortie (une ligne par tête) :
        x y w h occlusion blur

    Args:
        path (str)        : chemin absolu du fichier de sortie à créer.
        anns (list[dict]) : annotations à sauvegarder.

    Returns:
        None
    """
    # Crée les dossiers intermédiaires si nécessaire (ex : gt_A/, gt_B/)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for a in anns:
            # Écriture d'une ligne : les six valeurs séparées par des espaces
            f.write(
                f"{a['x']} {a['y']} {a['w']} {a['h']} "
                f"{a['occlusion']} {a['blur']}\n"
            )


def _save_links_file(path: str, links: list[tuple[int, int]]) -> None:
    """
    Sauvegarde les liens de correspondance entre têtes de Frame A et Frame B.

    Format de sortie (une ligne par lien) :
        idx_A  idx_B

    où idx_A est le numéro de ligne dans gt_A/*.txt et idx_B dans gt_B/*.txt.
    Les têtes dans gt_A sans lien sont des "left", celles dans gt_B sans lien
    sont des "entered".

    Args:
        path (str)                      : chemin absolu du fichier de sortie.
        links (list[tuple[int, int]])   : liste de paires (idx_A, idx_B).

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for idx_a, idx_b in links:
            f.write(f"{idx_a} {idx_b}\n")


# =============================================================================
# Simulation des artefacts vidéo
# =============================================================================

def _apply_video_artifacts(
    image_bgr: np.ndarray,
    rng: np.random.Generator,
    brightness_jitter: float = 0.08,
    color_jitter: float = 0.05,
    noise_sigma: float = 3.0,
) -> np.ndarray:
    """
    Applique de légers artefacts visuels à une image pour simuler une frame vidéo.

    Dans une vraie vidéo, chaque frame subit plusieurs dégradations physiques :
      - Le capteur de la caméra ajoute un bruit électronique aléatoire.
      - L'exposition automatique fait varier légèrement la luminosité.
      - La balance des blancs peut dériver, créant une légère dominante colorée.
      - La compression vidéo (H.264, HEVC…) introduit des artefacts de bloc.

    Cette fonction reproduit ces effets de façon contrôlée et subtile, avec
    des paramètres indépendants pour chaque frame (Frame A ≠ Frame B), ce qui
    est essentiel pour que le modèle apprenne à être robuste à ces variations.

    Pipeline d'effets appliqués (dans l'ordre) :
        1. Variation de luminosité globale   → simule l'exposition automatique
        2. Décalage colorimétrique par canal → simule la balance des blancs
        3. Bruit Gaussien additif            → simule le bruit du capteur

    Les trois effets sont subtils individuellement mais combinés, ils donnent
    aux deux frames un aspect suffisamment différent pour tromper un modèle
    naïf, l'obligeant à apprendre des caractéristiques robustes.

    Args:
        image_bgr (np.ndarray):
            Image au format BGR (OpenCV), dtype uint8, valeurs dans [0, 255].
            Shape attendue : (H, W, 3).

        rng (np.random.Generator):
            Générateur aléatoire NumPy partagé.  Chaque appel consomme quelques
            tirages aléatoires, ce qui garantit la reproductibilité si la même
            graine est utilisée.

        brightness_jitter (float):
            Amplitude maximale de la variation de luminosité globale.
            Un facteur b ∈ [1 - jitter, 1 + jitter] est tiré uniformément.
            Exemple : 0.08 → b ∈ [0.92, 1.08] → l'image peut être ±8 % plus lumineuse.
            Valeur par défaut : 0.08

        color_jitter (float):
            Amplitude maximale du décalage colorimétrique par canal BGR.
            Chaque canal reçoit un facteur indépendant c ∈ [1 - jitter, 1 + jitter].
            Les trois facteurs différents créent une légère dominante colorée.
            Exemple : 0.05 → canal bleu × 0.97, vert × 1.03, rouge × 1.01.
            Valeur par défaut : 0.05

        noise_sigma (float):
            Écart-type du bruit Gaussien additif, sur l'échelle de valeurs [0, 255].
            Un bruit N(0, sigma) est ajouté à chaque pixel de chaque canal.
            Exemple : 3.0 → bruit ≈ ±6 niveaux de gris, imperceptible à l'œil.
            Valeur par défaut : 3.0

    Returns:
        np.ndarray : image modifiée, même shape que l'entrée, dtype uint8.
    """
    # Conversion en float32 pour les opérations arithmétiques sans perte de précision
    # (les opérations sur uint8 provoqueraient des débordements silencieux)
    img = image_bgr.astype(np.float32)

    # --- Effet 1 : variation de luminosité globale ---
    # On multiplie TOUS les pixels de TOUS les canaux par un même facteur b.
    # b > 1 → image plus claire,  b < 1 → image plus sombre.
    # rng.uniform(a, b) tire un flottant aléatoire uniformément dans [a, b].
    b = float(rng.uniform(1.0 - brightness_jitter, 1.0 + brightness_jitter))
    img *= b

    # --- Effet 2 : décalage colorimétrique par canal ---
    # On multiplie chaque canal (B=0, G=1, R=2) par un facteur DIFFÉRENT.
    # Cela simule une balance des blancs imparfaite ou une dominante colorée.
    # Exemple : si le canal B reçoit ×0.97 et R reçoit ×1.04, l'image
    # paraît légèrement plus chaude (plus rouge, moins bleue).
    for canal in range(3):   # 0=Bleu, 1=Vert, 2=Rouge (ordre BGR d'OpenCV)
        c = float(rng.uniform(1.0 - color_jitter, 1.0 + color_jitter))
        img[:, :, canal] *= c

    # --- Effet 3 : bruit Gaussien additif ---
    # rng.normal(moyenne, sigma, shape) génère un tableau de bruit aléatoire.
    # On l'ajoute pixel par pixel à l'image.
    # Le bruit est indépendant pour chaque pixel ET pour chaque canal,
    # ce qui simule le bruit électronique d'un capteur photo.
    bruit = rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    img += bruit

    # --- Clipping et reconversion en uint8 ---
    # np.clip(x, 0, 255) ramène les valeurs hors de [0, 255] dans cet intervalle.
    # Sans ce clamp, les pixels trop clairs deviendraient > 255 (débordement),
    # et les pixels trop sombres deviendraient < 0.
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)

    return img


# =============================================================================
# Traitement d'une image complète
# =============================================================================

def _process_image(
    img_id: str,
    img_path: str,
    gt_path: str,
    label_info: dict,
    split_out_dir: str,
    shift_range: int,
    brightness_jitter: float,
    color_jitter: float,
    noise_sigma: float,
    rng: np.random.Generator,
) -> dict | None:
    """
    Traite une image du dataset original et génère tous les fichiers de la paire.

    Cette fonction orchestre l'ensemble du pipeline pour une image :
        1. Chargement de l'image avec OpenCV
        2. Calcul d'un décalage aléatoire (dx, dy)
        3. Découpage en Frame A et Frame B (deux crops complémentaires)
        4. Application d'artefacts vidéo indépendants sur chaque frame
        5. Génération des annotations pour chaque frame et des liens
        6. Sauvegarde des deux images JPEG, des deux fichiers GT et du fichier links

    Les artefacts sont appliqués APRÈS le découpage et avec des tirages
    aléatoires INDÉPENDANTS pour Frame A et Frame B, de sorte que les deux
    frames d'une même paire ont des conditions visuelles légèrement différentes
    — exactement comme deux instants successifs d'une vraie vidéo.

    Args:
        img_id (str)       : identifiant de l'image (ex : "0001").
        img_path (str)     : chemin vers l'image .jpg originale.
        gt_path (str)      : chemin vers le fichier d'annotations .txt original.
        label_info (dict)  : métadonnées de l'image (count, scene, weather, distractor).
        split_out_dir (str): dossier racine du split de sortie (ex : data_tracking/train/).
        shift_range (int)  : décalage maximum en pixels entre les deux crops.
        brightness_jitter (float) : amplitude max de variation de luminosité (±).
        color_jitter (float)      : amplitude max du décalage par canal BGR (±).
        noise_sigma (float)       : écart-type du bruit Gaussien additif.
        rng (np.random.Generator) : générateur aléatoire partagé.

    Returns:
        dict : métadonnées de la paire générée, à écrire dans image_labels.txt.
               Contient : id, count_A, count_B, count_stayed, count_left,
               count_entered, scene, weather, distractor.
        None : si l'image ne peut pas être lue (fichier corrompu ou absent).
    """
    # --- Chargement de l'image ---
    # cv2.imread charge en BGR (Blue-Green-Red), format natif d'OpenCV
    image = cv2.imread(img_path)
    if image is None:
        # L'image est absente ou corrompue : on signale et on passe
        print(f"  [AVERTISSEMENT] Impossible de lire : {img_path}")
        return None

    orig_h, orig_w = image.shape[:2]  # shape = (hauteur, largeur, canaux)

    # --- Calcul du décalage aléatoire ---
    dx, dy = _compute_shift(orig_w, orig_h, shift_range, rng)

    # --- Création des deux crops ---
    # Frame A = coin supérieur-gauche → lignes 0 à orig_h-dy, colonnes 0 à orig_w-dx
    # L'indexation numpy [y1:y2, x1:x2] sélectionne les pixels dans cet rectangle.
    # .copy() est nécessaire pour que le tableau résultant soit une vraie copie
    # indépendante (et non une simple "vue" de l'image originale) — sinon les
    # modifications des artefacts se répercuteraient sur les deux frames.
    frame_A = image[0: orig_h - dy,  0: orig_w - dx].copy()

    # Frame B = coin inférieur-droit → lignes dy à orig_h, colonnes dx à orig_w
    # Les deux crops ont exactement la même taille : (orig_h-dy) × (orig_w-dx)
    frame_B = image[dy: orig_h,      dx: orig_w].copy()

    # --- Application des artefacts vidéo ---
    # Chaque frame reçoit des paramètres aléatoires INDÉPENDANTS.
    # Le rng est partagé entre les deux appels : le premier appel consomme
    # quelques tirages, le second en consomme d'autres → valeurs différentes.
    # Résultat : Frame A et Frame B ont des conditions visuelles distinctes,
    # comme deux vraies frames consécutives d'une vidéo.
    frame_A = _apply_video_artifacts(
        frame_A, rng,
        brightness_jitter=brightness_jitter,
        color_jitter=color_jitter,
        noise_sigma=noise_sigma,
    )
    frame_B = _apply_video_artifacts(
        frame_B, rng,
        brightness_jitter=brightness_jitter,
        color_jitter=color_jitter,
        noise_sigma=noise_sigma,
    )

    # --- Génération des annotations ---
    anns = _parse_gt_file(gt_path)
    gt_A, gt_B, links = _generate_pair(anns, dx, dy, orig_w, orig_h)

    # --- Calcul des statistiques de tracking ---
    count_stayed  = len(links)
    count_A       = len(gt_A)
    count_B       = len(gt_B)
    count_left    = count_A - count_stayed   # dans A mais pas dans B
    count_entered = count_B - count_stayed   # dans B mais pas dans A

    # --- Sauvegarde des images ---
    # JPEG qualité 95 : bon équilibre entre taille fichier et fidélité visuelle
    img_A_path = os.path.join(split_out_dir, "frames_A", f"{img_id}.jpg")
    img_B_path = os.path.join(split_out_dir, "frames_B", f"{img_id}.jpg")
    os.makedirs(os.path.dirname(img_A_path), exist_ok=True)
    os.makedirs(os.path.dirname(img_B_path), exist_ok=True)
    cv2.imwrite(img_A_path, frame_A, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(img_B_path, frame_B, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # --- Sauvegarde des annotations GT ---
    _save_gt_file(
        os.path.join(split_out_dir, "gt_A", f"{img_id}.txt"), gt_A
    )
    _save_gt_file(
        os.path.join(split_out_dir, "gt_B", f"{img_id}.txt"), gt_B
    )

    # --- Sauvegarde des liens de correspondance ---
    _save_links_file(
        os.path.join(split_out_dir, "gt_links", f"{img_id}.txt"), links
    )

    # Retourne les métadonnées pour image_labels.txt
    return {
        "id":             img_id,
        "count_A":        count_A,
        "count_B":        count_B,
        "count_stayed":   count_stayed,
        "count_left":     count_left,
        "count_entered":  count_entered,
        "scene":          label_info["scene"],
        "weather":        label_info["weather"],
        "distractor":     label_info["distractor"],
    }


# =============================================================================
# Fonction principale — traitement de tous les splits
# =============================================================================

def build_tracking_dataset(
    data_root: str,
    output_root: str,
    splits: tuple[str, ...] = ("train", "val", "test"),
    shift_range: int = 30,
    brightness_jitter: float = 0.08,
    color_jitter: float = 0.05,
    noise_sigma: float = 3.0,
    seed: int = 42,
) -> None:
    """
    Génère le dataset de tracking complet à partir du dataset JHU-CROWD++.

    Pour chaque split ("train", "val", "test"), cette fonction :
        1. Lit les métadonnées (image_labels.txt)
        2. Pour chaque image, appelle ``_process_image`` qui génère la paire
        3. Écrit le nouveau image_labels.txt dans le dossier de sortie
        4. Affiche des statistiques résumées

    Arborescence générée sous ``output_root/`` :
        {train,val,test}/
            frames_A/         ← images Frame A (.jpg, avec artefacts vidéo)
            frames_B/         ← images Frame B (.jpg, artefacts indépendants de A)
            gt_A/             ← annotations Frame A (.txt)
            gt_B/             ← annotations Frame B (.txt)
            gt_links/         ← liens de correspondance (.txt)
            image_labels.txt  ← métadonnées tracking

    Note : les images sont sauvegardées à leur résolution originale (moins
    le crop shift).  Le redimensionnement final se fera dans dataGathering.py
    via le paramètre ``img_size`` du Dataset PyTorch.

    Args:
        data_root (str)   : chemin vers le dossier ``data/`` original.
        output_root (str) : chemin vers le dossier de sortie ``data_tracking/``.
        splits (tuple)    : liste des splits à traiter.
        shift_range (int) : décalage maximum en pixels entre les deux crops.
            Valeur par défaut 30 px. Représente un "pan" de caméra d'environ
            2 à 5 % de la dimension de l'image selon sa résolution.
        brightness_jitter (float) : amplitude max de variation de luminosité (±).
            Valeur par défaut 0.08 → variation de ±8 % autour de la luminosité d'origine.
            Augmenter pour des transitions lumineuses plus marquées (ex : 0.15).
        color_jitter (float) : amplitude max du décalage colorimétrique par canal (±).
            Valeur par défaut 0.05 → chaque canal BGR peut dériver de ±5 %.
            Augmenter pour des dominantes de couleur plus visibles (ex : 0.10).
        noise_sigma (float) : écart-type du bruit Gaussien additif sur [0-255].
            Valeur par défaut 3.0 → bruit fin, imperceptible mais mesurable.
            Augmenter pour simuler des conditions de faible lumière (ex : 6.0).
        seed (int)        : graine du générateur aléatoire.
            Même seed → même dataset généré (reproductibilité).

    Returns:
        None
    """
    # Générateur aléatoire avec graine fixe pour la reproductibilité
    # np.random.default_rng est plus moderne que np.random.seed()
    rng = np.random.default_rng(seed)

    for split in splits:
        print(f"\n{'='*60}")
        print(f"  Split : {split.upper()}")
        print(f"{'='*60}")

        # --- Chemins des données sources ---
        images_dir   = os.path.join(data_root, split, "images")
        gt_dir       = os.path.join(data_root, split, "gt")
        labels_file  = os.path.join(data_root, split, "image_labels.txt")

        # --- Dossier de sortie pour ce split ---
        split_out_dir = os.path.join(output_root, split)
        os.makedirs(split_out_dir, exist_ok=True)

        # --- Lecture des métadonnées sources ---
        labels  = _parse_image_labels(labels_file)
        img_ids = list(labels.keys())
        print(f"  {len(img_ids)} images à traiter...")

        results: list[dict] = []
        errors = 0

        for i, img_id in enumerate(img_ids):
            # Affichage de la progression toutes les 200 images
            if (i + 1) % 200 == 0 or (i + 1) == len(img_ids):
                print(f"  [{i + 1:>4}/{len(img_ids)}] ...")

            img_path = os.path.join(images_dir, f"{img_id}.jpg")
            gt_path  = os.path.join(gt_dir,     f"{img_id}.txt")

            result = _process_image(
                img_id            = img_id,
                img_path          = img_path,
                gt_path           = gt_path,
                label_info        = labels[img_id],
                split_out_dir     = split_out_dir,
                shift_range       = shift_range,
                brightness_jitter = brightness_jitter,
                color_jitter      = color_jitter,
                noise_sigma       = noise_sigma,
                rng               = rng,
            )

            if result is not None:
                results.append(result)
            else:
                errors += 1

        # --- Écriture du fichier image_labels.txt tracking ---
        labels_out_path = os.path.join(split_out_dir, "image_labels.txt")
        with open(labels_out_path, "w") as f:
            for r in results:
                # Format : id,count_A,count_B,count_stayed,count_left,count_entered,scene,weather,distractor
                f.write(
                    f"{r['id']},"
                    f"{r['count_A']},{r['count_B']},"
                    f"{r['count_stayed']},{r['count_left']},{r['count_entered']},"
                    f"{r['scene']},{r['weather']},{r['distractor']}\n"
                )

        # --- Statistiques résumées du split ---
        if results:
            stayed_total  = sum(r["count_stayed"]  for r in results)
            left_total    = sum(r["count_left"]    for r in results)
            entered_total = sum(r["count_entered"] for r in results)
            a_total       = sum(r["count_A"]       for r in results)

            print(f"\n  Résultats {split.upper()} :")
            print(f"    Paires générées : {len(results):>6} / {len(img_ids)}")
            print(f"    Erreurs         : {errors:>6}")
            print(f"    Têtes (Frame A) : {a_total:>8,}")
            print(f"    — stayed        : {stayed_total:>8,}  ({100*stayed_total/a_total:.1f} %)")
            print(f"    — left          : {left_total:>8,}  ({100*left_total/a_total:.1f} %)")
            print(f"    — entered       : {entered_total:>8,}")

    print(f"\n\nDataset de tracking généré dans : {output_root}")
    print("Utilisez dataGathering.JHUCrowdTrackingDataset pour le charger dans PyTorch.")


# =============================================================================
# Point d'entrée — lancement direct du script
# =============================================================================

if __name__ == "__main__":
    # Calcul des chemins relatifs à ce fichier (projects/dataCleaning.py)
    # os.path.dirname(__file__) → dossier projects/
    # ".."                      → racine du projet
    DATA_ROOT   = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
    OUTPUT_ROOT = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data_tracking")
    )

    print(f"Source      : {DATA_ROOT}")
    print(f"Destination : {OUTPUT_ROOT}")

    build_tracking_dataset(
        data_root         = DATA_ROOT,
        output_root       = OUTPUT_ROOT,
        splits            = ("train", "val", "test"),
        shift_range       = 30,   # décalage max de 30 px entre Frame A et Frame B
        brightness_jitter = 0.08, # luminosité ±8 % entre les deux frames
        color_jitter      = 0.05, # dominante colorée ±5 % par canal BGR
        noise_sigma       = 3.0,  # bruit capteur : écart-type 3 niveaux sur 255
        seed              = 42,   # graine fixe → dataset reproductible
    )
