"""
dataGathering.py — Chargement des données JHU-CROWD++ v2.0 pour PyTorch
========================================================================

Ce module fournit tout ce qu'il faut pour lire le dataset JHU-CROWD++ et
l'exposer à un réseau de neurones PyTorch.  Il gère deux cas d'usage :

  1. Comptage de foule (generate_density=True)
     L'image est accompagnée d'une "density map" : une carte 2-D dont
     la somme des pixels est égale au nombre de personnes dans l'image.
     C'est la cible que le réseau apprendra à prédire.

  2. Détection / tracking (generate_density=False)
     L'image est accompagnée des boîtes englobantes (bounding boxes) de
     chaque tête annotée.  Utile pour un détecteur de type YOLO/DETR ou
     pour un tracker multi-objets.

Structure attendue sous `data_root/` :
    train/
        images/          ← fichiers .jpg
        gt/              ← un .txt par image (annotations tête par tête)
        image_labels.txt ← métadonnées globales par image
    val/   (même structure)
    test/  (même structure)

Format du fichier gt/*.txt (une ligne = une tête) :
    x  y  w  h  occlusion  blur
    x, y  : coordonnées du centre de la tête (en pixels)
    w, h  : largeur / hauteur approximative de la tête
    occlusion : 1=visible  2=partiellement occluse  3=totalement occluse
    blur      : 0=nette    1=floue

Format du fichier image_labels.txt (une ligne = une image) :
    id, count, scene, weather, distractor
    id         : identifiant numérique (ex : "0001")
    count      : nombre total de personnes dans l'image
    scene      : type de lieu ("stadium", "street", "concert"…)
    weather    : 0=clair  1=brouillard  2=pluie  3=neige
    distractor : 0=image valide  1=image à ignorer (distracteur)

Dépendances Python :
    torch, torchvision, numpy, scipy, Pillow
"""

import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from dataCleaning import (
    _compute_shift,
    _generate_pair,
    _compute_warp,
    _generate_pair_warp,
    _apply_video_artifacts,
)


# =============================================================================
# Classe principale : JHUCrowdDataset
# =============================================================================

class JHUCrowdDataset(Dataset):
    """
    Dataset PyTorch pour le jeu de données JHU-CROWD++ v2.0.

    Cette classe hérite de `torch.utils.data.Dataset`, ce qui est le
    contrat standard de PyTorch pour fournir des données à un DataLoader.
    Il suffit d'implémenter trois méthodes :
        __init__  → initialisation / lecture des métadonnées
        __len__   → nombre total d'exemples disponibles
        __getitem__ → retourne un exemple à partir de son indice

    Deux modes de fonctionnement sont disponibles selon l'usage :
        generate_density=True  → mode comptage  (density map)
        generate_density=False → mode tracking  (bounding boxes)

    Attributs de classe (constantes partagées par toutes les instances) :
        WEATHER   : dictionnaire de décodage des codes météo
        OCCLUSION : dictionnaire de décodage des niveaux d'occlusion
    """

    # Constantes de décodage des valeurs numériques stockées dans les fichiers
    WEATHER = {0: "no-degradation", 1: "fog/haze", 2: "rain", 3: "snow"}
    OCCLUSION = {1: "visible", 2: "partial", 3: "full"}

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: tuple | None = None,
        augment: bool = False,
        generate_density: bool = True,
        density_sigma: float = 15.0,
    ):
        """
        Initialise le dataset et charge les métadonnées en mémoire.

        Seules les métadonnées (image_labels.txt) sont lues ici.
        Les images et annotations sont chargées à la demande dans
        __getitem__ pour ne pas surcharger la RAM.

        Args:
            root_dir (str):
                Chemin vers le dossier `data/` qui contient les sous-dossiers
                train/, val/ et test/.
                Exemple : "d:/jhu_crowd_v2.0/SytemeIntelligentRobinAlexis/data"

            split (str):
                Sous-ensemble à utiliser : "train", "val" ou "test".
                - "train" : 2 272 images, utilisées pour entraîner le modèle.
                - "val"   : 500 images,   utilisées pour régler les hyperparamètres.
                - "test"  : 1 600 images, utilisées pour évaluer les performances finales.

            img_size (tuple | None):
                Taille cible des images sous la forme (hauteur, largeur) en pixels,
                par exemple (512, 512).  Si None, les images gardent leur taille
                d'origine (tailles variables selon les photos).

            augment (bool):
                Si True, applique des transformations aléatoires pour enrichir
                artificiellement le dataset (flip horizontal + changements de couleur).
                N'a d'effet que pour le split "train" ; ignoré pour val/test.

            generate_density (bool):
                True  → mode comptage : __getitem__ renvoie une density map.
                False → mode tracking : __getitem__ renvoie des bounding boxes.

            density_sigma (float):
                Écart-type (σ) du filtre Gaussien utilisé pour générer la density
                map.  Plus σ est grand, plus les "taches" autour de chaque tête
                sont étalées.  Valeur typique : 10 à 20 pixels.
        """
        # Vérification que le nom du split est valide
        assert split in ("train", "val", "test"), (
            f"split '{split}' inconnu — choisir parmi 'train', 'val', 'test'."
        )

        # Sauvegarde des paramètres comme attributs d'instance
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.generate_density = generate_density
        self.density_sigma = density_sigma

        # L'augmentation n'est activée que pour le split d'entraînement
        # (on ne veut pas perturber les données de validation/test)
        self.augment = augment and split == "train"

        # Construction des chemins vers les trois sous-dossiers du split
        self.images_dir = os.path.join(root_dir, split, "images")  # .jpg
        self.gt_dir = os.path.join(root_dir, split, "gt")          # .txt annotations
        self.labels_file = os.path.join(root_dir, split, "image_labels.txt")

        # Lecture du fichier image_labels.txt → dictionnaire {id_image: métadonnées}
        # Ce dictionnaire reste en RAM pendant toute la durée de l'entraînement
        self.image_labels = self._load_image_labels()

        # Liste ordonnée des identifiants d'images (ex : ["0001", "0008", ...])
        # C'est cette liste qui sert d'index : image_ids[i] est l'image numéro i
        self.image_ids = list(self.image_labels.keys())

        # Normalisation ImageNet : soustrait la moyenne et divise par l'écart-type
        # de chaque canal RGB, calculés sur ImageNet.  C'est une pratique standard
        # qui améliore la convergence car les poids pré-entraînés attendent des
        # images dans cette plage de valeurs.
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # moyenne R, G, B sur ImageNet
            std=[0.229, 0.224, 0.225],   # écart-type R, G, B sur ImageNet
        )

    # -------------------------------------------------------------------------
    # Méthodes privées de parsing (lecture des fichiers texte)
    # -------------------------------------------------------------------------

    def _load_image_labels(self) -> dict:
        """
        Lit et parse le fichier `image_labels.txt` du split courant.

        Chaque ligne du fichier a le format :
            id,count,scene,weather,distractor
        Exemple :
            0001,161,water park,0,0

        Returns:
            dict[str, dict] : dictionnaire dont les clés sont les identifiants
            d'images (chaîne de caractères, ex : "0001") et les valeurs sont
            des dictionnaires contenant :
                - "count"      (int)  : nombre de personnes dans l'image
                - "scene"      (str)  : type de lieu ("stadium", "street"…)
                - "weather"    (int)  : code météo (0=clair, 1=brouillard…)
                - "distractor" (int)  : 1 si l'image est à ignorer, 0 sinon
        """
        labels = {}

        with open(self.labels_file, "r") as f:
            for line in f:
                # Supprime les espaces/retours à la ligne en début et fin,
                # puis découpe la ligne sur les virgules
                parts = [p.strip() for p in line.strip().split(",")]

                # Ignore les lignes mal formées (moins de 5 colonnes)
                if len(parts) < 5:
                    continue

                # parts[0] = id image, parts[1] = count, etc.
                img_id = parts[0]
                labels[img_id] = {
                    "count": int(parts[1]),   # conversion texte → entier
                    "scene": parts[2],        # gardé comme texte
                    "weather": int(parts[3]),
                    "distractor": int(parts[4]),
                }

        return labels

    def _load_annotations(self, img_id: str) -> list[dict]:
        """
        Lit le fichier d'annotation tête par tête (`gt/<img_id>.txt`).

        Chaque ligne du fichier décrit une tête annotée avec le format :
            x  y  w  h  occlusion  blur
        Exemple :
            106 114 24 25 1 0

        Args:
            img_id (str): identifiant de l'image (ex : "0001").

        Returns:
            list[dict] : liste de dictionnaires, un par tête annotée.
            Chaque dictionnaire contient :
                - "x"         (int) : colonne du centre de la tête (pixel)
                - "y"         (int) : ligne du centre de la tête (pixel)
                - "w"         (int) : largeur approximative de la tête (pixels)
                - "h"         (int) : hauteur approximative de la tête (pixels)
                - "occlusion" (int) : 1=visible, 2=partielle, 3=totale
                - "blur"      (int) : 0=nette, 1=floue
            Retourne une liste vide si le fichier n'existe pas.
        """
        gt_path = os.path.join(self.gt_dir, f"{img_id}.txt")
        anns = []

        # Certaines images peuvent ne pas avoir de fichier gt (foule vide)
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                for line in f:
                    # Découpe sur les espaces (séparateur entre colonnes)
                    parts = line.strip().split()

                    # Ignore les lignes incomplètes
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

    # -------------------------------------------------------------------------
    # Génération de la density map
    # -------------------------------------------------------------------------

    def _make_density_map(
        self,
        orig_h: int,
        orig_w: int,
        anns: list[dict],
        out_h: int,
        out_w: int,
    ) -> np.ndarray:
        """
        Génère une density map 2-D à partir des positions des têtes annotées.

        Principe :
            1. On crée une image noire (zéros) à la taille de l'image originale.
            2. Pour chaque tête annotée, on place un "1" au pixel correspondant
               au centre de cette tête.
            3. On applique un filtre Gaussien qui "étale" chaque point en une
               tache floue (comme une empreinte de doigt sur du papier).
            4. Si on a redimensionné l'image, on redimensionne aussi la carte
               et on renormalise pour que la somme totale reste inchangée.

        Pourquoi un filtre Gaussien ?
            Une carte de points discrets est difficile à apprendre pour un
            réseau (signal très épars).  L'étalement Gaussien crée une cible
            "douce" et continue.  La propriété clé est que l'intégrale (somme
            des pixels) de la carte est égale au nombre de personnes — le réseau
            apprend donc à prédire une carte dont la somme donne le comptage.

        Args:
            orig_h (int) : hauteur de l'image originale avant resize (en pixels).
            orig_w (int) : largeur de l'image originale avant resize (en pixels).
            anns (list[dict]) : liste des annotations retournée par _load_annotations.
            out_h (int) : hauteur de sortie souhaitée (après resize éventuel).
            out_w (int) : largeur de sortie souhaitée (après resize éventuel).

        Returns:
            np.ndarray de forme (out_h, out_w) et dtype float32.
            La somme de tous les pixels est égale au nombre de têtes annotées.
        """
        # --- Étape 1 : carte d'impulsitons (points discrets) ----------------
        # Tableau de zéros de la taille de l'image originale
        dm = np.zeros((orig_h, orig_w), dtype=np.float32)

        for a in anns:
            x, y = a["x"], a["y"]
            # Vérifie que le point est bien à l'intérieur de l'image
            # (quelques annotations peuvent déborder légèrement)
            if 0 <= x < orig_w and 0 <= y < orig_h:
                # On place 1.0 au pixel (y, x) — note : numpy indexe [ligne, colonne]
                dm[y, x] += 1.0

        # --- Étape 2 : lissage Gaussien ------------------------------------
        # gaussian_filter remplace chaque pic par une tache Gaussienne de largeur σ.
        # Résultat : une carte 2-D continue dont l'intégrale ≈ nombre de têtes.
        dm = gaussian_filter(dm, sigma=self.density_sigma)

        # --- Étape 3 : redimensionnement (si nécessaire) -------------------
        if (out_h, out_w) != (orig_h, orig_w):
            # Sauvegarde de la somme avant resize (= nombre de têtes)
            total = dm.sum()

            # PIL.Image.resize attend (largeur, hauteur) et travaille sur uint8/float32.
            # BILINEAR : interpolation bilinéaire — bon compromis précision/vitesse.
            dm_pil = Image.fromarray(dm).resize((out_w, out_h), Image.BILINEAR)
            dm = np.array(dm_pil, dtype=np.float32)

            # Renormalisation : le resize peut modifier légèrement la somme totale
            # à cause de l'interpolation.  On la remet à sa valeur d'origine.
            if dm.sum() > 0:
                dm *= total / dm.sum()

        return dm

    # -------------------------------------------------------------------------
    # Augmentation des données
    # -------------------------------------------------------------------------

    def _apply_augmentation(
        self, image: Image.Image, anns: list[dict]
    ) -> tuple[Image.Image, list[dict]]:
        """
        Applique un flip horizontal aléatoire à l'image et aux annotations.

        L'augmentation de données consiste à créer des variantes artificielles
        des images d'entraînement pour que le modèle voie plus de diversité
        sans avoir besoin de collecter de nouvelles données.

        Pourquoi flip uniquement ?
            Un flip vertical changerait la perspective de façon non réaliste
            (les têtes seraient en bas).  Le flip horizontal est neutre : une
            foule vue de gauche ou de droite est identique.

        Important : quand on retourne l'image, il faut aussi mettre à jour
        les coordonnées x de chaque annotation, sinon elles pointent vers le
        mauvais endroit.  La formule est : x_nouveau = largeur - 1 - x_ancien.

        Args:
            image (PIL.Image.Image) : image à transformer.
            anns (list[dict]) : annotations correspondantes (modifiées en cohérence).

        Returns:
            tuple (PIL.Image.Image, list[dict]) :
                - image éventuellement retournée
                - annotations avec coordonnées x mises à jour
        """
        # Tirage aléatoire : 50 % de chance d'appliquer le flip
        if np.random.rand() < 0.5:
            w = image.width  # largeur originale, nécessaire pour le calcul

            # Retournement horizontal de l'image
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Mise à jour de la coordonnée x de chaque annotation.
            # {**a, "x": ...} crée une copie du dictionnaire a en remplaçant "x".
            anns = [{**a, "x": w - 1 - a["x"]} for a in anns]

        return image, anns

    # -------------------------------------------------------------------------
    # Interface PyTorch Dataset (méthodes obligatoires)
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Retourne le nombre total d'images dans ce split.

        Méthode requise par PyTorch : le DataLoader appelle len(dataset)
        pour savoir combien d'exemples sont disponibles et construire les
        indices de chaque batch.

        Returns:
            int : nombre d'images (ex : 2272 pour "train").
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Charge et retourne un exemple complet à partir de son indice.

        Méthode requise par PyTorch : le DataLoader appelle dataset[i] pour
        chaque exemple à inclure dans un batch.  Tout le travail lourd
        (lecture du .jpg, parsing du .txt, génération de la density map…)
        est fait ici, à la demande.

        Le retour dépend du mode choisi à la construction :

        Mode comptage (generate_density=True) :
            {
              "image"       : Tensor (3, H, W)  — image normalisée
              "density_map" : Tensor (1, H, W)  — carte dont la somme = count
              "count"       : int               — nombre de personnes (vérité terrain)
              "image_id"    : str               — identifiant de l'image
            }

        Mode tracking (generate_density=False) :
            {
              "image"      : Tensor (3, H, W)     — image normalisée
              "boxes"      : Tensor (N, 4)         — N boîtes [x1, y1, x2, y2]
              "occlusion"  : Tensor (N,)           — niveau d'occlusion par tête
              "blur"       : Tensor (N,)           — netteté par tête
              "count"      : int                   — nombre de personnes
              "scene"      : str                   — type de lieu
              "weather"    : int                   — code météo
              "distractor" : int                   — 1 si image distracteur
              "image_id"   : str                   — identifiant de l'image
            }

        Args:
            idx (int) : indice de l'exemple (entre 0 et len(dataset)-1).

        Returns:
            dict : voir les deux formats ci-dessus.
        """
        # Récupération de l'identifiant et des métadonnées globales de l'image
        img_id = self.image_ids[idx]
        label_info = self.image_labels[img_id]

        # --- Chargement de l'image ------------------------------------------
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        # convert("RGB") garantit 3 canaux même si l'image source est en niveaux
        # de gris ou en RGBA
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size  # PIL renvoie (largeur, hauteur)

        # --- Chargement des annotations (positions des têtes) ---------------
        anns = self._load_annotations(img_id)

        # --- Augmentation aléatoire (entraînement uniquement) ---------------
        if self.augment:
            image, anns = self._apply_augmentation(image, anns)

        # --- Redimensionnement de l'image -----------------------------------
        if self.img_size is not None:
            out_h, out_w = self.img_size
            # PIL.Image.resize prend (largeur, hauteur) — ordre inverse de img_size
            image = image.resize((out_w, out_h), Image.BILINEAR)
        else:
            # Pas de resize : on garde la taille originale
            out_h, out_w = orig_h, orig_w

        # --- Jitter de couleur (entraînement uniquement) --------------------
        # Modifie légèrement luminosité, contraste et saturation pour que le
        # réseau soit robuste aux variations d'éclairage
        if self.augment:
            image = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            )(image)

        # --- Conversion PIL Image → Tensor PyTorch --------------------------
        # transforms.ToTensor() : (H, W, 3) uint8 [0-255] → (3, H, W) float32 [0-1]
        # self.normalize()      : soustrait la moyenne ImageNet, divise par std
        img_tensor = self.normalize(transforms.ToTensor()(image))

        # ====================================================================
        # Mode comptage : génération de la density map
        # ====================================================================
        if self.generate_density:
            # Génération de la carte 2-D (numpy array float32)
            dm = self._make_density_map(orig_h, orig_w, anns, out_h, out_w)

            # Conversion numpy → Tensor et ajout d'une dimension de canal :
            # (H, W) → (1, H, W)  pour être cohérent avec le format (C, H, W) de PyTorch
            density_tensor = torch.from_numpy(dm).unsqueeze(0)

            return {
                "image": img_tensor,            # Tensor (3, H, W) — image RGB normalisée
                "density_map": density_tensor,  # Tensor (1, H, W) — somme des pixels = count
                "count": label_info["count"],   # int — vérité terrain (pour évaluation)
                "image_id": img_id,             # str — utile pour les logs et visualisations
            }

        # ====================================================================
        # Mode tracking : bounding boxes redimensionnées
        # ====================================================================

        # Facteurs d'échelle pour passer des coordonnées originales aux
        # coordonnées dans l'image redimensionnée
        sx = out_w / orig_w  # facteur horizontal
        sy = out_h / orig_h  # facteur vertical

        if anns:
            # Conversion des annotations (centre + taille) en boîtes [x1,y1,x2,y2]
            # Format "xyxy" (coin supérieur-gauche, coin inférieur-droit) — standard
            # utilisé par la plupart des détecteurs (Faster-RCNN, DETR, YOLO…)
            boxes = torch.tensor(
                [
                    [
                        (a["x"] - a["w"] / 2) * sx,  # x1 = bord gauche
                        (a["y"] - a["h"] / 2) * sy,  # y1 = bord haut
                        (a["x"] + a["w"] / 2) * sx,  # x2 = bord droit
                        (a["y"] + a["h"] / 2) * sy,  # y2 = bord bas
                    ]
                    for a in anns
                ],
                dtype=torch.float32,
            )
            # Vecteurs d'attributs supplémentaires par tête
            occlusion = torch.tensor([a["occlusion"] for a in anns], dtype=torch.long)
            blur = torch.tensor([a["blur"] for a in anns], dtype=torch.long)
        else:
            # Image sans aucune annotation (foule vide) : tenseurs vides de bonne forme
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            occlusion = torch.zeros(0, dtype=torch.long)
            blur = torch.zeros(0, dtype=torch.long)

        return {
            "image": img_tensor,              # Tensor (3, H, W)
            "boxes": boxes,                   # Tensor (N, 4) — N boîtes englobantes
            "occlusion": occlusion,           # Tensor (N,)   — 1=visible 2=partiel 3=total
            "blur": blur,                     # Tensor (N,)   — 0=net 1=flou
            "count": label_info["count"],     # int
            "scene": label_info["scene"],     # str
            "weather": label_info["weather"], # int (0=clair … 3=neige)
            "distractor": label_info["distractor"],  # int (0 ou 1)
            "image_id": img_id,               # str
        }


# =============================================================================
# Fonction de collate personnalisée pour le mode tracking
# =============================================================================

def _collate_tracking(batch: list[dict]) -> dict:
    """
    Regroupe une liste d'exemples individuels en un seul batch.

    Le DataLoader appelle automatiquement cette fonction pour "empiler"
    les exemples d'un batch en tenseurs.  La fonction par défaut de PyTorch
    suppose que tous les tenseurs ont la même taille — ce qui est vrai pour
    les images (3×H×W) mais pas pour les boîtes englobantes (le nombre N
    de têtes varie d'une image à l'autre).

    Cette fonction laisse donc les boîtes, occlusions et flous sous forme
    de listes Python (une entrée par image dans le batch) plutôt que de
    tenter de les empiler en un seul tenseur.

    Args:
        batch (list[dict]) : liste de dictionnaires, un par exemple.
            Chaque dict a les clés retournées par __getitem__ en mode tracking.

    Returns:
        dict : même structure que les dicts d'entrée, mais avec :
            - "image"  → tenseur (batch_size, 3, H, W)
            - "count"  → tenseur (batch_size,)
            - les autres champs → listes Python de longueur batch_size
    """
    return {
        # torch.stack empile les tenseurs d'images le long d'une nouvelle dim 0
        "image": torch.stack([b["image"] for b in batch]),  # (B, 3, H, W)

        # Les boîtes ont un nombre variable de lignes → on garde une liste
        "boxes": [b["boxes"] for b in batch],         # liste de Tensor (Ni, 4)
        "occlusion": [b["occlusion"] for b in batch], # liste de Tensor (Ni,)
        "blur": [b["blur"] for b in batch],           # liste de Tensor (Ni,)

        # Les scalaires sont regroupés en un tenseur 1-D
        "count": torch.tensor([b["count"] for b in batch]),  # (B,)

        # Les chaînes/entiers restent en listes Python
        "scene": [b["scene"] for b in batch],
        "weather": [b["weather"] for b in batch],
        "distractor": [b["distractor"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }


# =============================================================================
# Fonction utilitaire : création d'un DataLoader prêt à l'emploi
# =============================================================================

def get_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 8,
    img_size: tuple | None = None,
    generate_density: bool = True,
    augment: bool = True,
    num_workers: int = 0,
    shuffle: bool | None = None,
) -> tuple[DataLoader, JHUCrowdDataset]:
    """
    Crée et retourne un DataLoader PyTorch pour un split donné.

    Un DataLoader est l'objet qui, à chaque itération de la boucle
    d'entraînement, fournit un nouveau batch d'images au modèle.
    Il gère automatiquement :
        - le découpage en batchs de taille `batch_size`
        - le mélange aléatoire (`shuffle`)
        - le chargement parallèle en arrière-plan (`num_workers`)
        - le transfert vers le GPU (`pin_memory`)

    Args:
        data_root (str):
            Chemin vers le dossier `data/`.

        split (str):
            "train", "val" ou "test".

        batch_size (int):
            Nombre d'images par batch.  Valeurs courantes : 4, 8, 16, 32.
            À ajuster selon la mémoire GPU disponible.

        img_size (tuple | None):
            (H, W) de redimensionnement, ou None pour garder les tailles d'origine.
            Attention : sans resize, les images ont des tailles différentes et
            le DataLoader ne peut pas les empiler → il faut fournir un img_size.

        generate_density (bool):
            True  → mode comptage (density maps).
            False → mode tracking (bounding boxes).

        augment (bool):
            Active l'augmentation aléatoire (flip + ColorJitter).
            N'a d'effet que pour split="train".

        num_workers (int):
            Nombre de processus parallèles pour charger les données.
            0 = chargement dans le processus principal (plus sûr sur Windows).
            2 ou 4 sur Linux/Mac pour accélérer le chargement.

        shuffle (bool | None):
            True  → ordre aléatoire à chaque époque (recommandé pour train).
            False → ordre fixe (recommandé pour val/test).
            None  → automatique : True si split=="train", False sinon.

    Returns:
        tuple[DataLoader, JHUCrowdDataset] :
            - DataLoader : à utiliser dans la boucle `for batch in loader`
            - JHUCrowdDataset : l'objet dataset sous-jacent (pour inspecter les données)
    """
    # Déduction automatique du shuffle si non spécifié
    if shuffle is None:
        shuffle = split == "train"

    # Création du dataset
    dataset = JHUCrowdDataset(
        root_dir=data_root,
        split=split,
        img_size=img_size,
        augment=augment,
        generate_density=generate_density,
    )

    # Création du DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # pin_memory=True alloue la RAM en "mémoire épinglée" (non paginable),
        # ce qui accélère le transfert CPU → GPU.  Activé seulement si GPU disponible.
        pin_memory=torch.cuda.is_available(),
        # En mode tracking, les batchs ont des boîtes de tailles variables →
        # on utilise notre collate personnalisée ; en mode comptage la collate
        # par défaut de PyTorch suffit (tous les tenseurs ont la même forme).
        collate_fn=None if generate_density else _collate_tracking,
    )

    return loader, dataset


# =============================================================================
# Fonction utilitaire : statistiques descriptives d'un split
# =============================================================================

def print_split_stats(data_root: str, split: str) -> None:
    """
    Affiche des statistiques descriptives sur un split du dataset.

    Lit uniquement le fichier image_labels.txt (pas les images ni les gt)
    pour calculer rapidement :
        - nombre d'images
        - min / max / moyenne / médiane du comptage de personnes
        - répartition des types de scènes
        - répartition des conditions météo

    Args:
        data_root (str) : chemin vers le dossier `data/`.
        split (str)     : "train", "val" ou "test".

    Returns:
        None (affiche dans la console).
    """
    labels_file = os.path.join(data_root, split, "image_labels.txt")

    counts = []                    # liste des comptages (un par image)
    scenes = {}                    # dictionnaire {nom_scene: nb_occurrences}
    weathers = {0: 0, 1: 0, 2: 0, 3: 0}  # compteur par code météo

    with open(labels_file, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 5:
                continue

            counts.append(int(parts[1]))

            # Comptage des scènes (dict.get avec default 0 pour initialisation)
            scene = parts[2]
            scenes[scene] = scenes.get(scene, 0) + 1

            # Comptage des codes météo
            weathers[int(parts[3])] += 1

    # Décodage des codes numériques pour l'affichage
    weather_labels = {0: "clear", 1: "fog/haze", 2: "rain", 3: "snow"}

    print(f"\n=== {split.upper()} ({len(counts)} images) ===")
    print(
        f"  Count  — min {min(counts):,}  max {max(counts):,}  "
        f"mean {np.mean(counts):.1f}  median {np.median(counts):.1f}"
    )
    # Tri des scènes par fréquence décroissante pour lisibilité
    print(f"  Scenes — {dict(sorted(scenes.items(), key=lambda kv: -kv[1]))}")
    print(f"  Weather— {dict((weather_labels[k], v) for k, v in weathers.items())}")


# =============================================================================
# Test rapide (exécuté uniquement si ce fichier est lancé directement)
# =============================================================================

# =============================================================================
# Dataset PyTorch pour le tracking — charge les données de data_tracking/
# =============================================================================

import torchvision.transforms.functional as TF  # API fonctionnelle pour les transforms


class JHUCrowdTrackingDataset(Dataset):
    """
    Dataset PyTorch pour le tracking de foule entre paires de frames.

    Ce dataset charge les données générées par ``dataCleaning.build_tracking_dataset``
    (dossier ``data_tracking/``).

    Pour chaque exemple, deux frames (A et B) sont retournées avec :
      - les boîtes englobantes des têtes dans chaque frame,
      - les liens de correspondance (quelle tête de A correspond à quelle tête de B).

    Ces informations permettent d'entraîner un modèle à :
        1. Détecter les têtes dans chaque frame
        2. Apparier les têtes entre frames (stayed / left / entered)
        3. Compter en temps réel les entrées, sorties et personnes présentes

    Contenu retourné par ``__getitem__`` :
        {
          "frame_A"       : Tensor (3, H, W)   — Frame A normalisée
          "frame_B"       : Tensor (3, H, W)   — Frame B normalisée
          "boxes_A"       : Tensor (N_A, 4)    — boîtes xyxy dans Frame A
          "boxes_B"       : Tensor (N_B, 4)    — boîtes xyxy dans Frame B
          "links"         : Tensor (L, 2)      — paires [idx_A, idx_B] des têtes appariées
          "count_A"       : int                — personnes dans Frame A
          "count_B"       : int                — personnes dans Frame B
          "count_stayed"  : int                — personnes présentes dans les deux frames
          "count_left"    : int                — personnes qui ont quitté le champ
          "count_entered" : int                — personnes qui sont entrées dans le champ
          "image_id"      : str                — identifiant de l'image source
          "scene"         : str                — type de lieu
          "weather"       : int                — code météo
        }
    """

    WEATHER = {0: "no-degradation", 1: "fog/haze", 2: "rain", 3: "snow"}

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: tuple | None = None,
        augment: bool = False,
        shift_range: int = 30,
        jitter_range: int = 15,
        brightness_jitter: float = 0.08,
        color_jitter_vid: float = 0.05,
        noise_sigma: float = 3.0,
    ):
        """
        Initialise le dataset de tracking.

        Les frames A et B sont générées À LA VOLÉE dans __getitem__ à partir
        des images originales JHU-CROWD++. Chaque epoch voit des paires
        différentes (shift aléatoire différent).

        Args:
            root_dir (str):
                Chemin vers le dossier ``data/`` original (JHU-CROWD++).
                (Anciennement data_tracking/ — plus besoin de dataCleaning.py)

            split (str):
                "train", "val" ou "test".

            img_size (tuple | None):
                (H, W) de redimensionnement. Fortement recommandé car les crops
                ont des tailles variables selon le shift tiré aléatoirement.

            augment (bool):
                Si True (train uniquement), applique un flip horizontal aléatoire
                et cohérent sur les deux frames.

            shift_range (int):
                Amplitude max du pan global entre Frame A et Frame B (défaut 30).

            jitter_range (int):
                Amplitude max du jitter par coin pour le warp géométrique
                (défaut 15, plafonné à 2 % de la dimension). Le décalage entre
                les deux frames varie alors selon la position dans l'image.

            brightness_jitter (float):
                Amplitude max de variation de luminosité par frame (±, défaut 0.08).

            color_jitter_vid (float):
                Amplitude max de décalage colorimétrique par canal (±, défaut 0.05).

            noise_sigma (float):
                Écart-type du bruit Gaussien additif (défaut 3.0).
        """
        assert split in ("train", "val", "test"), f"Split inconnu : {split}"

        self.root_dir          = root_dir
        self.split             = split
        self.img_size          = img_size
        self.augment           = augment and split == "train"
        self.shift_range       = shift_range
        self.jitter_range      = jitter_range
        self.brightness_jitter = brightness_jitter
        self.color_jitter_vid  = color_jitter_vid
        self.noise_sigma       = noise_sigma

        # Chemins vers les données originales JHU-CROWD++
        self.images_dir  = os.path.join(root_dir, split, "images")
        self.gt_dir      = os.path.join(root_dir, split, "gt")
        self.labels_file = os.path.join(root_dir, split, "image_labels.txt")

        # Lecture des métadonnées (image_labels.txt original — 5 colonnes)
        self.image_labels = self._load_image_labels()
        self.image_ids    = list(self.image_labels.keys())

        # Normalisation ImageNet standard
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    # -------------------------------------------------------------------------
    # Méthodes privées de parsing
    # -------------------------------------------------------------------------

    def _load_image_labels(self) -> dict:
        """
        Lit le fichier ``image_labels.txt`` original JHU-CROWD++ (5 colonnes).

        Format de chaque ligne :
            id,count,scene,weather,distractor
        Exemple :
            0001,161,water park,0,0

        Returns:
            dict[str, dict] avec les champs "count", "scene", "weather", "distractor".
        """
        labels = {}
        with open(self.labels_file, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 5:
                    continue
                labels[parts[0]] = {
                    "count":      int(parts[1]),
                    "scene":      parts[2],
                    "weather":    int(parts[3]),
                    "distractor": int(parts[4]),
                }
        return labels

    def _load_gt_file(self, gt_path: str) -> list[dict]:
        """
        Lit un fichier d'annotations (gt_A/*.txt ou gt_B/*.txt).

        Format identique au dataset original :
            x  y  w  h  occlusion  blur

        Args:
            gt_path (str) : chemin vers le fichier .txt.

        Returns:
            list[dict] : liste des têtes annotées (peut être vide).
        """
        anns = []
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
                        "blur":      int(parts[5]),
                    })
        return anns

    def _load_links_file(self, links_path: str) -> list[tuple[int, int]]:
        """
        Lit le fichier de liens de correspondance (gt_links/*.txt).

        Format : une ligne par lien → idx_A  idx_B
        où idx_A est le numéro de ligne dans gt_A et idx_B dans gt_B.

        Args:
            links_path (str) : chemin vers le fichier .txt des liens.

        Returns:
            list[tuple[int,int]] : liste de paires (idx_A, idx_B).
        """
        links = []
        if not os.path.exists(links_path):
            return links
        with open(links_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    links.append((int(parts[0]), int(parts[1])))
        return links

    # -------------------------------------------------------------------------
    # Conversion annotations → boîtes englobantes
    # -------------------------------------------------------------------------

    def _anns_to_boxes(
        self,
        anns: list[dict],
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> torch.Tensor:
        """
        Convertit les annotations (centre + taille) en boîtes xyxy mises à l'échelle.

        Les annotations stockent la position du CENTRE de chaque tête (x, y) et
        sa taille approximative (w, h).  Le format "xyxy" (coin supérieur-gauche,
        coin inférieur-droit) est le standard utilisé par la plupart des détecteurs.

        Formule :
            x1 = (x - w/2) * scale_x
            y1 = (y - h/2) * scale_y
            x2 = (x + w/2) * scale_x
            y2 = (y + h/2) * scale_y

        Args:
            anns (list[dict]) : annotations avec clés x, y, w, h.
            scale_x (float)   : facteur d'échelle horizontal (si image redimensionnée).
            scale_y (float)   : facteur d'échelle vertical.

        Returns:
            Tensor de forme (N, 4) dtype float32, ou (0, 4) si liste vide.
        """
        if not anns:
            return torch.zeros((0, 4), dtype=torch.float32)

        return torch.tensor(
            [
                [
                    (a["x"] - a["w"] / 2) * scale_x,   # x1 = bord gauche
                    (a["y"] - a["h"] / 2) * scale_y,   # y1 = bord haut
                    (a["x"] + a["w"] / 2) * scale_x,   # x2 = bord droit
                    (a["y"] + a["h"] / 2) * scale_y,   # y2 = bord bas
                ]
                for a in anns
            ],
            dtype=torch.float32,
        )

    # -------------------------------------------------------------------------
    # Augmentation cohérente sur les deux frames
    # -------------------------------------------------------------------------

    def _apply_augmentation(
        self,
        img_A: Image.Image,
        img_B: Image.Image,
        anns_A: list[dict],
        anns_B: list[dict],
    ) -> tuple:
        """
        Applique des transformations aléatoires COHÉRENTES aux deux frames.

        Cohérence = même décision aléatoire pour Frame A et Frame B, afin que
        les correspondances spatiales entre les deux frames restent valides après
        augmentation.

        Transformations appliquées :
          1. Flip horizontal (50 % de probabilité) :
               - La décision flip / ne-flip est la même pour les deux frames.
               - Les coordonnées x sont mises à jour : x_new = w - 1 - x_old
               - Les indices des liens (links) ne changent pas (ce sont juste
                 des numéros de lignes dans gt_A et gt_B, pas des coordonnées).

          2. Perturbation de couleur (brightness / contrast / saturation) :
               - Les MÊMES paramètres aléatoires sont appliqués aux deux frames.
               - Simule des conditions d'éclairage légèrement variables mais
                 cohérentes entre deux instants proches.

        Args:
            img_A, img_B   : images PIL des deux frames.
            anns_A, anns_B : listes d'annotations des deux frames.

        Returns:
            tuple (img_A, img_B, anns_A, anns_B) potentiellement transformés.
        """
        # --- 1. Flip horizontal ---
        if np.random.rand() < 0.5:
            # Largeur des deux frames (elles ont la même taille)
            w = img_A.width

            # Retournement des deux images
            img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)

            # Mise à jour des coordonnées x : formule x_new = w - 1 - x_old
            # {**a, "x": ...} crée une copie du dict a avec x mis à jour
            anns_A = [{**a, "x": w - 1 - a["x"]} for a in anns_A]
            anns_B = [{**a, "x": w - 1 - a["x"]} for a in anns_B]
            # Note : les liens (idx_A, idx_B) ne dépendent pas des coordonnées,
            # ils restent valides après le flip.

        return img_A, img_B, anns_A, anns_B

    # -------------------------------------------------------------------------
    # Interface PyTorch Dataset (méthodes obligatoires)
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Retourne le nombre de paires de frames dans ce split.

        Returns:
            int : nombre d'exemples (= nombre d'images originales du split).
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Charge et retourne une paire complète (Frame A + Frame B) avec ses labels.

        Pipeline complet :
            1. Chargement des deux images JPEG
            2. Chargement des annotations gt_A, gt_B et des liens gt_links
            3. Augmentation (flip + couleur) si mode entraînement
            4. Redimensionnement si img_size est fourni
            5. Conversion PIL → Tensor normalisé
            6. Conversion annotations → boîtes xyxy (avec mise à l'échelle)
            7. Conversion liens → Tensor

        Args:
            idx (int) : indice de l'exemple (entre 0 et len(dataset)-1).

        Returns:
            dict avec les clés :
                "frame_A"       : Tensor (3, H, W) — Frame A normalisée
                "frame_B"       : Tensor (3, H, W) — Frame B normalisée
                "boxes_A"       : Tensor (N_A, 4)  — boîtes xyxy Frame A
                "boxes_B"       : Tensor (N_B, 4)  — boîtes xyxy Frame B
                "links"         : Tensor (L, 2)    — paires [idx_A, idx_B]
                "count_A"       : int
                "count_B"       : int
                "count_stayed"  : int
                "count_left"    : int
                "count_entered" : int
                "image_id"      : str
                "scene"         : str
                "weather"       : int
        """
        img_id     = self.image_ids[idx]
        label_info = self.image_labels[img_id]

        # --- Chargement de l'image originale (BGR) ---
        img_path  = os.path.join(self.images_dir, f"{img_id}.jpg")
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image introuvable : {img_path}")
        orig_h, orig_w = image_bgr.shape[:2]

        # --- Génération du warp aléatoire à la volée ---
        # Chaque appel produit un warp différent → diversité entre époques.
        # Pan global + jitter par coin : le décalage entre les deux frames
        # varie selon la position dans l'image (cf. dataCleaning._compute_warp).
        rng = np.random.default_rng()
        H_warp, margin = _compute_warp(
            orig_w, orig_h, self.shift_range, self.jitter_range, rng
        )

        # --- Création des deux frames (même crop central) ---
        frame_A_bgr = image_bgr[margin:orig_h - margin, margin:orig_w - margin].copy()
        warped_bgr  = cv2.warpPerspective(image_bgr, H_warp, (orig_w, orig_h))
        frame_B_bgr = warped_bgr[margin:orig_h - margin, margin:orig_w - margin].copy()

        # --- Artefacts vidéo indépendants (bruit, luminosité, couleur) ---
        frame_A_bgr = _apply_video_artifacts(
            frame_A_bgr, rng, self.brightness_jitter, self.color_jitter_vid, self.noise_sigma
        )
        frame_B_bgr = _apply_video_artifacts(
            frame_B_bgr, rng, self.brightness_jitter, self.color_jitter_vid, self.noise_sigma
        )

        # --- Génération des annotations à la volée ---
        anns  = self._load_gt_file(os.path.join(self.gt_dir, f"{img_id}.txt"))
        gt_A, gt_B, links = _generate_pair_warp(anns, H_warp, margin, orig_w, orig_h)

        # --- BGR → PIL RGB (pour flip et transforms PyTorch) ---
        img_A = Image.fromarray(cv2.cvtColor(frame_A_bgr, cv2.COLOR_BGR2RGB))
        img_B = Image.fromarray(cv2.cvtColor(frame_B_bgr, cv2.COLOR_BGR2RGB))

        # --- Flip horizontal aléatoire cohérent (entraînement uniquement) ---
        if self.augment:
            img_A, img_B, gt_A, gt_B = self._apply_augmentation(
                img_A, img_B, gt_A, gt_B
            )

        # --- Redimensionnement ---
        crop_w, crop_h = orig_w - 2 * margin, orig_h - 2 * margin
        if self.img_size is not None:
            out_h, out_w = self.img_size
            img_A = img_A.resize((out_w, out_h), Image.BILINEAR)
            img_B = img_B.resize((out_w, out_h), Image.BILINEAR)
        else:
            out_h, out_w = crop_h, crop_w

        # --- Facteurs d'échelle pour les coordonnées des boîtes ---
        scale_x = out_w / crop_w
        scale_y = out_h / crop_h

        # --- Conversion PIL → Tensor normalisé ---
        tensor_A = self.normalize(transforms.ToTensor()(img_A))
        tensor_B = self.normalize(transforms.ToTensor()(img_B))

        # --- Annotations → boîtes xyxy mises à l'échelle ---
        boxes_A = self._anns_to_boxes(gt_A, scale_x, scale_y)
        boxes_B = self._anns_to_boxes(gt_B, scale_x, scale_y)

        # --- Liens → Tensor (L, 2) ---
        if links:
            links_tensor = torch.tensor(links, dtype=torch.long)
        else:
            links_tensor = torch.zeros((0, 2), dtype=torch.long)

        count_stayed  = len(links)
        count_A       = len(gt_A)
        count_B       = len(gt_B)

        return {
            "frame_A":       tensor_A,
            "frame_B":       tensor_B,
            "boxes_A":       boxes_A,
            "boxes_B":       boxes_B,
            "links":         links_tensor,
            "count_A":       count_A,
            "count_B":       count_B,
            "count_stayed":  count_stayed,
            "count_left":    count_A - count_stayed,
            "count_entered": count_B - count_stayed,
            "image_id":      img_id,
            "scene":         label_info["scene"],
            "weather":       label_info["weather"],
        }


# =============================================================================
# Collate personnalisée pour les paires de frames (tailles variables)
# =============================================================================

def _collate_tracking_pairs(batch: list[dict]) -> dict:
    """
    Regroupe une liste de paires en un seul batch pour le DataLoader.

    Problème : les boîtes (boxes_A, boxes_B) et les liens (links) ont un
    nombre variable de lignes selon l'image (le nombre de têtes diffère).
    PyTorch ne peut pas empiler automatiquement des tenseurs de tailles
    différentes → on garde ces champs sous forme de listes Python.

    Les images, en revanche, ont toutes la même taille si ``img_size`` est
    défini → on peut les empiler en un seul tenseur (B, 3, H, W).

    Args:
        batch (list[dict]) : liste de dictionnaires retournés par __getitem__.

    Returns:
        dict : même structure que __getitem__, mais :
            - "frame_A", "frame_B" → Tensor (B, 3, H, W)
            - "count_*"            → Tensor (B,)
            - "boxes_A", "boxes_B", "links" → listes de longueur B
            - "scene", "weather", "image_id" → listes de longueur B
    """
    return {
        # Les images ont la même taille → on peut les empiler avec torch.stack
        "frame_A": torch.stack([b["frame_A"] for b in batch]),   # (B, 3, H, W)
        "frame_B": torch.stack([b["frame_B"] for b in batch]),   # (B, 3, H, W)

        # Les boîtes et liens ont des tailles variables → on garde des listes
        # Chaque élément de la liste est un Tensor (Ni, 4) ou (Li, 2)
        "boxes_A":  [b["boxes_A"]  for b in batch],  # liste de Tensor (N_Ai, 4)
        "boxes_B":  [b["boxes_B"]  for b in batch],  # liste de Tensor (N_Bi, 4)
        "links":    [b["links"]    for b in batch],  # liste de Tensor (Li, 2)

        # Les comptages sont des scalaires → on les empile en tenseurs 1-D
        "count_A":       torch.tensor([b["count_A"]       for b in batch]),
        "count_B":       torch.tensor([b["count_B"]       for b in batch]),
        "count_stayed":  torch.tensor([b["count_stayed"]  for b in batch]),
        "count_left":    torch.tensor([b["count_left"]    for b in batch]),
        "count_entered": torch.tensor([b["count_entered"] for b in batch]),

        # Les chaînes et entiers restent en listes Python
        "image_id": [b["image_id"] for b in batch],
        "scene":    [b["scene"]    for b in batch],
        "weather":  [b["weather"]  for b in batch],
    }


# =============================================================================
# Fonction utilitaire : DataLoader de tracking prêt à l'emploi
# =============================================================================

def get_tracking_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 4,
    img_size: tuple | None = None,
    augment: bool = True,
    num_workers: int = 0,
    shuffle: bool | None = None,
    shift_range: int = 30,
    jitter_range: int = 15,
    brightness_jitter: float = 0.08,
    color_jitter_vid: float = 0.05,
    noise_sigma: float = 3.0,
) -> tuple[DataLoader, JHUCrowdTrackingDataset]:
    """
    Crée et retourne un DataLoader PyTorch pour le dataset de tracking.

    Ce DataLoader charge les paires (Frame A, Frame B) avec leurs annotations
    et liens, prêtes à être utilisées dans une boucle d'entraînement.

    Exemple d'utilisation dans une boucle d'entraînement :
        loader, dataset = get_tracking_dataloader("data_tracking/", split="train",
                                                   batch_size=4, img_size=(512,512))
        for batch in loader:
            frame_A  = batch["frame_A"]   # Tensor (4, 3, 512, 512)
            frame_B  = batch["frame_B"]   # Tensor (4, 3, 512, 512)
            boxes_A  = batch["boxes_A"]   # liste de 4 tenseurs (Ni, 4)
            links    = batch["links"]     # liste de 4 tenseurs (Li, 2)
            # ... passer dans le modèle

    Args:
        data_root (str):
            Chemin vers le dossier ``data_tracking/`` généré par dataCleaning.py.

        split (str):
            "train", "val" ou "test".

        batch_size (int):
            Nombre de paires par batch.  Valeurs courantes : 2, 4, 8.
            Les paires contiennent deux images → la mémoire GPU requise est
            environ le double d'un dataset de comptage pour le même batch_size.

        img_size (tuple | None):
            (H, W) de redimensionnement.  Fortement recommandé car les crops
            ont des tailles légèrement variables selon le shift calculé.

        augment (bool):
            Active flip + perturbation de couleur cohérente (train seulement).

        num_workers (int):
            Processus parallèles de chargement.  0 = sûr sur Windows.

        shuffle (bool | None):
            True = ordre aléatoire (défaut pour train), False = ordre fixe.

    Returns:
        tuple[DataLoader, JHUCrowdTrackingDataset] :
            - DataLoader : à utiliser dans ``for batch in loader:``
            - JHUCrowdTrackingDataset : le dataset sous-jacent
    """
    if shuffle is None:
        shuffle = split == "train"

    dataset = JHUCrowdTrackingDataset(
        root_dir          = data_root,
        split             = split,
        img_size          = img_size,
        augment           = augment,
        shift_range       = shift_range,
        jitter_range      = jitter_range,
        brightness_jitter = brightness_jitter,
        color_jitter_vid  = color_jitter_vid,
        noise_sigma       = noise_sigma,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        # pin_memory=True accélère le transfert CPU→GPU (si GPU disponible)
        pin_memory  = torch.cuda.is_available(),
        # Les boîtes et liens ont des tailles variables → collate personnalisée
        collate_fn  = _collate_tracking_pairs,
    )

    return loader, dataset


# =============================================================================
# Statistiques rapides du dataset de tracking
# =============================================================================

def print_tracking_stats(data_root: str, split: str) -> None:
    """
    Affiche des statistiques descriptives sur un split du dataset de tracking.

    Lit uniquement le fichier image_labels.txt (pas les images) pour calculer :
        - nombre de paires
        - distribution des stayed / left / entered
        - pourcentage de flux entrant/sortant

    Args:
        data_root (str) : chemin vers le dossier ``data_tracking/``.
        split (str)     : "train", "val" ou "test".

    Returns:
        None (affiche dans la console).
    """
    labels_file = os.path.join(data_root, split, "image_labels.txt")

    # Accumulateurs pour les statistiques globales
    counts_A, stayed_list, left_list, entered_list = [], [], [], []

    with open(labels_file, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 9:
                continue
            counts_A.append(int(parts[1]))
            stayed_list.append(int(parts[3]))
            left_list.append(int(parts[4]))
            entered_list.append(int(parts[5]))

    if not counts_A:
        print(f"  Aucune donnée dans {labels_file}")
        return

    total_A       = sum(counts_A)
    total_stayed  = sum(stayed_list)
    total_left    = sum(left_list)
    total_entered = sum(entered_list)

    print(f"\n=== TRACKING {split.upper()} ({len(counts_A)} paires) ===")
    print(f"  Têtes Frame A  — min {min(counts_A):,}  max {max(counts_A):,}  "
          f"mean {np.mean(counts_A):.1f}")
    print(f"  Stayed   : {total_stayed:>8,}  ({100*total_stayed/total_A:.1f} % de Frame A)")
    print(f"  Left     : {total_left:>8,}  ({100*total_left/total_A:.1f} % de Frame A)")
    print(f"  Entered  : {total_entered:>8,}")


# =============================================================================
# Smoke test (existant — inchangé)
# =============================================================================

if __name__ == "__main__":
    # Calcul du chemin vers data/ relatif à ce fichier (projects/dataGathering.py)
    # os.path.dirname(__file__) → dossier projects/
    # ".."                      → remonte d'un niveau → racine du projet
    # "data"                    → dossier data/
    DATA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    print(f"Data root: {DATA_ROOT}")

    # Affichage des statistiques pour les 3 splits
    for split in ("train", "val", "test"):
        print_split_stats(DATA_ROOT, split)

    # --- Test 1 : mode comptage (density map) ---
    print("\n--- Mode comptage (density map) ---")
    ds = JHUCrowdDataset(
        DATA_ROOT,
        split="train",
        img_size=(512, 512),   # toutes les images redimensionnées en 512×512
        generate_density=True,
        augment=False,
    )
    s = ds[0]  # charge le premier exemple
    print(f"  image       : {s['image'].shape}")        # doit être (3, 512, 512)
    print(f"  density_map : {s['density_map'].shape}  "
          f"sum={s['density_map'].sum():.1f}  gt={s['count']}")
    # La somme de la density map doit être très proche du comptage ground truth

    # --- Test 2 : mode tracking (bounding boxes) ---
    print("\n--- Mode tracking (bounding boxes) ---")
    ds_t = JHUCrowdDataset(
        DATA_ROOT,
        split="train",
        img_size=(512, 512),
        generate_density=False,
        augment=False,
    )
    s_t = ds_t[0]
    print(f"  image  : {s_t['image'].shape}")            # (3, 512, 512)
    print(f"  boxes  : {s_t['boxes'].shape}  (N têtes × [x1,y1,x2,y2])")
    print(f"  count={s_t['count']}  scene={s_t['scene']}  "
          f"weather={JHUCrowdDataset.WEATHER[s_t['weather']]}")

    # --- Test 3 : DataLoader (batch de 4 images) ---
    print("\n--- DataLoader (batch_size=4, mode comptage) ---")
    loader, _ = get_dataloader(
        DATA_ROOT,
        split="train",
        batch_size=4,
        img_size=(512, 512),
        generate_density=True,
        augment=True,
        num_workers=12,  # 0 = pas de multiprocessing (plus sûr sur Windows)
    )
    # next(iter(loader)) récupère le premier batch
    batch = next(iter(loader))
    print(f"  images       : {batch['image'].shape}")       # (4, 3, 512, 512)
    print(f"  density_maps : {batch['density_map'].shape}") # (4, 1, 512, 512)
    print(f"  counts       : {batch['count'].tolist()}")    # liste de 4 entiers