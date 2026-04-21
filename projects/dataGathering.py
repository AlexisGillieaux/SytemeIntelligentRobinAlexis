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

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
        num_workers=0,  # 0 = pas de multiprocessing (plus sûr sur Windows)
    )
    # next(iter(loader)) récupère le premier batch
    batch = next(iter(loader))
    print(f"  images       : {batch['image'].shape}")       # (4, 3, 512, 512)
    print(f"  density_maps : {batch['density_map'].shape}") # (4, 1, 512, 512)
    print(f"  counts       : {batch['count'].tolist()}")    # liste de 4 entiers