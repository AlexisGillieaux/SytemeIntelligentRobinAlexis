"""
model.py — Réseau de tracking de foule (CrowdTrackingNet)
==========================================================

Problème à résoudre :
    Étant donné deux frames consécutives (Frame A et Frame B), le modèle doit
    produire une carte mono-canal qui encode trois types d'événements :

        0   → arrière-plan (aucun événement de tracking)
      255   → pixel appartenant au trait reliant une tête « stayed » (A → B)
       +1   → position d'une tête « entered » (apparue dans Frame B)
       -1   → position de sortie d'une tête « left »  (quittée après Frame A)

    En résumé : le modèle répond à la question
    "que s'est-il passé entre Frame A et Frame B, pixel par pixel ?"

Architecture choisie : U-Net (encodeur-décodeur avec skip connections)
    • Entrée  : Frame A et Frame B concaténées → (B, 6, H, W)
    • Sortie  : carte de tracking              → (B, 1, H, W)

    Pourquoi U-Net ?
        Le SimpleCNN original était un classifieur global (une classe pour toute
        l'image).  Ici, la tâche est de la prédiction DENSE (on doit annoter
        chaque pixel).  U-Net est l'architecture standard pour ce type de
        tâche : l'encodeur extrait les features, le décodeur les reprojette
        à la résolution d'origine grâce aux skip connections.

Données utilisées :
    Générées par dataCleaning.build_tracking_dataset → data_tracking/
    Chargées via dataGathering.get_tracking_dataloader

Dépendances :
    torch, numpy, cv2, dataGathering
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Import du DataLoader depuis dataGathering.py (même dossier)
sys.path.insert(0, os.path.dirname(__file__))
from dataGathering import get_tracking_dataloader


# =============================================================================
# Génération de la carte cible (target map) à la volée
# =============================================================================

def make_target_map(
    gt_A: list[dict],
    gt_B: list[dict],
    links: list,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """
    Génère la carte cible mono-canal pour une paire (Frame A, Frame B).

    Cette fonction est appelée à chaque batch pendant l'entraînement.
    Elle produit la supervision du modèle sans stocker quoi que ce soit
    sur disque — la carte est calculée à la volée depuis les annotations.

    Valeurs encodées dans la carte :
        0   → arrière-plan (la quasi-totalité des pixels)
      255   → pixel sur le trait d'une tête « stayed » (trajet A → B)
       +1   → position d'une tête « entered » (point unique)
       -1   → position de sortie d'une tête « left » (point clampé au bord)

    Args:
        gt_A (list[dict])  : annotations Frame A — chaque dict a les clés x, y, w, h.
        gt_B (list[dict])  : annotations Frame B — même format.
        links (list)       : paires [[idx_A, idx_B], …] des têtes appariées.
        img_h (int)        : hauteur de Frame B en pixels.
        img_w (int)        : largeur de Frame B en pixels.

    Returns:
        Tensor (1, img_h, img_w) float32 — canal unique, valeurs {-1, 0, 1, 255}.
    """
    # Tableau numpy de zéros (fond = 0 par défaut)
    target = np.zeros((img_h, img_w), dtype=np.float32)

    # Déduction du décalage global (dx, dy) depuis le premier lien disponible.
    # Relation : x_A = x_B + dx  →  dx = x_A - x_B
    if links:
        iA0, iB0 = links[0][0], links[0][1]
        dx = gt_A[iA0]["x"] - gt_B[iB0]["x"]
        dy = gt_A[iA0]["y"] - gt_B[iB0]["y"]
    else:
        dx, dy = 0, 0

    stayed_A_idx = {int(pair[0]) for pair in links}
    stayed_B_idx = {int(pair[1]) for pair in links}

    # ---- Stayed : trait blanc (255) de la position A vers la position B ----
    # Le trait est dessiné dans l'espace pixel de Frame B.
    # Point de départ (position A en coords Frame B) : (x_B + dx, y_B + dy)
    # Point d'arrivée (position actuelle)            : (x_B, y_B)
    for pair in links:
        iB = int(pair[1])
        x_B, y_B = gt_B[iB]["x"], gt_B[iB]["y"]
        cv2.line(target,
                 (x_B + dx, y_B + dy),   # position précédente (Frame A)
                 (x_B,      y_B),         # position actuelle   (Frame B)
                 color=255.0, thickness=1)

    # ---- Entered : point +1 à la position de la tête dans Frame B ----
    for i, a in enumerate(gt_B):
        if i in stayed_B_idx:
            continue
        x, y = a["x"], a["y"]
        if 0 <= x < img_w and 0 <= y < img_h:
            target[y, x] = 1.0

    # ---- Left : point -1 à la position de sortie clampée dans Frame B ----
    # (x_A - dx) peut être négatif si la tête est sortie par le bord gauche.
    # np.clip la ramène dans [0, dimension - 1].
    for i, a in enumerate(gt_A):
        if i in stayed_A_idx:
            continue
        x_exit = int(np.clip(a["x"] - dx, 0, img_w - 1))
        y_exit = int(np.clip(a["y"] - dy, 0, img_h - 1))
        target[y_exit, x_exit] = -1.0

    # (H, W) numpy → (1, H, W) Tensor PyTorch
    return torch.from_numpy(target).unsqueeze(0)


def build_target_batch(
    batch: dict,
    img_h: int,
    img_w: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Génère les cartes cibles pour tout un batch.

    Le DataLoader retourne les boîtes au format xyxy (x1,y1,x2,y2).
    Cette fonction reconstruit les centres (x,y) depuis ces boîtes pour
    alimenter make_target_map, puis empile les résultats en un tenseur batch.

    Args:
        batch  : dict retourné par _collate_tracking_pairs (dataGathering.py).
                 Doit contenir "boxes_A", "boxes_B", "links".
        img_h  : hauteur des images (après redimensionnement).
        img_w  : largeur des images.
        device : device sur lequel placer le tenseur de sortie.

    Returns:
        Tensor (B, 1, img_h, img_w) float32.
    """

    def boxes_to_anns(boxes: torch.Tensor) -> list[dict]:
        """Reconstruit des dicts {x,y,w,h} à partir de boîtes xyxy."""
        anns = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            anns.append({
                "x": int((x1 + x2) / 2),
                "y": int((y1 + y2) / 2),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
            })
        return anns

    targets = []
    for i in range(len(batch["boxes_A"])):
        gt_A  = boxes_to_anns(batch["boxes_A"][i])
        gt_B  = boxes_to_anns(batch["boxes_B"][i])
        links = batch["links"][i].tolist()   # Tensor (L,2) → list of [iA, iB]

        targets.append(make_target_map(gt_A, gt_B, links, img_h, img_w))

    return torch.stack(targets).to(device)   # (B, 1, H, W)


# =============================================================================
# Blocs de construction du réseau
# =============================================================================

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Bloc double convolution : Conv→BN→ReLU→Conv→BN→ReLU.

    C'est la brique de base du U-Net.  Deux convolutions successives
    augmentent le champ réceptif.  La Batch Normalization (BN) stabilise
    l'entraînement et réduit la dépendance au learning rate.
    padding=1 préserve la taille spatiale H×W à chaque convolution.

    Args:
        in_ch  : canaux d'entrée.
        out_ch : canaux de sortie.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# =============================================================================
# Architecture principale : CrowdTrackingNet
# =============================================================================

class CrowdTrackingNet(nn.Module):
    """
    Réseau encodeur-décodeur (U-Net) pour le tracking de foule entre deux frames.

    Architecture :

        Input (B, 6, H, W)           ← Frame A + Frame B concaténées
             │
        ┌────▼────┐  enc1 (32)   ──────────────────────────────────┐ skip 1
        │  pool   │                                                  │
        ├────▼────┤  enc2 (64)   ─────────────────────────────┐ skip 2
        │  pool   │                                             │
        ├────▼────┤  enc3 (128)  ────────────────────────┐ skip 3
        │  pool   │                                       │
        ├────▼────┤  enc4 (256)  ───────────────────┐ skip 4
        │  pool   │                                  │
        ├────▼────┤  bottleneck (512)                │
        │  up4    │──────────────────────── cat skip4 ┘
        ├────▼────┤  dec4 (256)
        │  up3    │───────────────────── cat skip3 ┘
        ├────▼────┤  dec3 (128)
        │  up2    │────────────────── cat skip2 ┘
        ├────▼────┤  dec2 (64)
        │  up1    │─────────────── cat skip1 ┘
        ├────▼────┤  dec1 (32)
        │  head   │  Conv 1×1 → 1 canal (pas d'activation)
        └────▼────┘
        Output (B, 1, H, W)          ← carte de tracking

    Pourquoi sans activation finale ?
        La sortie doit couvrir {-1, 0, 1, 255}.  Une sigmoid sortirait [0,1],
        une tanh sortirait [-1,1] — aucune ne peut atteindre 255.
        On laisse donc la convolution finale linéaire et on laisse la perte
        guider le réseau vers les bonnes valeurs.
    """

    def __init__(self, base_ch: int = 32):
        """
        Args:
            base_ch (int) :
                Nombre de filtres au premier niveau de l'encodeur.
                Les niveaux suivants ont 2×, 4×, 8×, 16× filtres.
                • base_ch=16 : ~2 M paramètres — entraînable sur CPU lent / GPU modeste.
                • base_ch=32 : ~8 M paramètres — recommandé (défaut).
                • base_ch=64 : ~32 M paramètres — GPU avec ≥ 8 Go VRAM.
        """
        super().__init__()

        # ---- Encodeur (chemin descendant) --------------------------------
        # MaxPool2d(2,2) divise H et W par 2 entre chaque niveau.
        self.enc1 = _conv_block(6,          base_ch)       # → (B,  32, H,    W)
        self.enc2 = _conv_block(base_ch,    base_ch * 2)   # → (B,  64, H/2,  W/2)
        self.enc3 = _conv_block(base_ch*2,  base_ch * 4)   # → (B, 128, H/4,  W/4)
        self.enc4 = _conv_block(base_ch*4,  base_ch * 8)   # → (B, 256, H/8,  W/8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- Goulot d'étranglement (bottleneck) --------------------------
        # Niveau le plus profond : champ réceptif maximal, représentation
        # la plus abstraite (relations globales entre les deux frames).
        self.bottleneck = _conv_block(base_ch*8, base_ch*16)  # → (B, 512, H/16, W/16)

        # ---- Décodeur (chemin ascendant) ---------------------------------
        # ConvTranspose2d : upsampling ×2 "appris" (contrairement à bilinear).
        # Après chaque upsampling, on concatène avec les features de l'encodeur
        # (skip connection) pour récupérer les détails spatiaux fins.

        self.up4  = nn.ConvTranspose2d(base_ch*16, base_ch*8,  kernel_size=2, stride=2)
        self.dec4 = _conv_block(base_ch*16, base_ch*8)   # 16× = up(8×) + skip(8×)

        self.up3  = nn.ConvTranspose2d(base_ch*8,  base_ch*4,  kernel_size=2, stride=2)
        self.dec3 = _conv_block(base_ch*8,  base_ch*4)

        self.up2  = nn.ConvTranspose2d(base_ch*4,  base_ch*2,  kernel_size=2, stride=2)
        self.dec2 = _conv_block(base_ch*4,  base_ch*2)

        self.up1  = nn.ConvTranspose2d(base_ch*2,  base_ch,    kernel_size=2, stride=2)
        self.dec1 = _conv_block(base_ch*2,  base_ch)

        # ---- Projection finale : N canaux → 1 canal ----------------------
        # kernel_size=1 : "convolution pointwise" — mélange les canaux sans
        # toucher au voisinage spatial.  Pas d'activation : sortie linéaire.
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du réseau.

        Args:
            x (Tensor) : (B, 6, H, W) — Frame A et Frame B concaténées.
                         Les 3 premiers canaux sont Frame A, les 3 suivants Frame B.

        Returns:
            Tensor (B, 1, H, W) — carte de tracking prédite.
        """
        # ---- Encodeur : extraction des features à plusieurs échelles ----
        e1 = self.enc1(x)                # (B,  32, H,    W)   — textures fines
        e2 = self.enc2(self.pool(e1))    # (B,  64, H/2,  W/2) — formes locales
        e3 = self.enc3(self.pool(e2))    # (B, 128, H/4,  W/4) — parties de corps
        e4 = self.enc4(self.pool(e3))    # (B, 256, H/8,  W/8) — objets entiers

        # ---- Bottleneck : représentation globale la plus compressée ----
        b  = self.bottleneck(self.pool(e4))  # (B, 512, H/16, W/16)

        # ---- Décodeur : reconstruction progressive avec skip connections ----
        # torch.cat(..., dim=1) fusionne sur la dimension canal
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))  # (B, 256, H/8, W/8)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 128, H/4, W/4)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B,  64, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B,  32, H,   W)

        return self.head(d1)   # (B, 1, H, W)


# =============================================================================
# Fonction de perte
# =============================================================================

def tracking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_event: float = 20.0,
) -> torch.Tensor:
    """
    MSE pondérée pour la carte de tracking.

    Problème d'équilibre de classes :
        La quasi-totalité des pixels sont 0 (arrière-plan).
        Une MSE classique apprendrait à tout prédire ~0 avec une faible perte,
        en ignorant complètement les événements (stayed/entered/left).

    Solution : les pixels d'événements (valeur ≠ 0) sont pondérés ×weight_event
    lors du calcul de la perte, forçant le réseau à les traiter prioritairement.

    Args:
        pred (Tensor)          : sortie du modèle (B, 1, H, W).
        target (Tensor)        : carte cible      (B, 1, H, W).
        weight_event (float)   : facteur de pondération pour les pixels non-nuls.
                                 Augmenter si le modèle ignore les événements rares.

    Returns:
        Tensor scalaire différentiable — valeur de la perte.
    """
    # Masque : 1 là où il y a un événement, 0 pour l'arrière-plan
    mask = (target != 0).float()

    # Poids par pixel : 1 (fond) ou weight_event (événement)
    weights = 1.0 + mask * (weight_event - 1.0)

    return (weights * (pred - target) ** 2).mean()


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    event_threshold: float = 0.3,
    stayed_threshold: float = 50.0,
) -> dict:
    """
    Calcule des métriques détaillées par type d'événement.

    Returns:
        dict avec les clés :
          mse_bg       — MSE sur les pixels de fond (target == 0)
          mse_events   — MSE sur les pixels d'événement (target != 0)
          recall_pts   — recall des têtes entered/left (|target| == 1)
                         = % de pixels d'événement où |pred| > event_threshold
          iou_stayed   — IoU des traits stayed (target == 255)
                         entre pred >= stayed_threshold et target == 255
    """
    with torch.no_grad():
        mask_bg     = (target == 0)
        mask_events = (target != 0)
        mask_pts    = (target.abs() == 1)    # entered (+1) et left (-1)
        mask_stayed = (target == 255)

        # MSE fond
        if mask_bg.any():
            mse_bg = ((pred[mask_bg] - target[mask_bg]) ** 2).mean().item()
        else:
            mse_bg = 0.0

        # MSE événements
        if mask_events.any():
            mse_events = ((pred[mask_events] - target[mask_events]) ** 2).mean().item()
        else:
            mse_events = 0.0

        # Recall entered/left : parmi les vrais pixels ±1, combien le modèle détecte ?
        if mask_pts.any():
            detected = (pred[mask_pts].abs() > event_threshold)
            recall_pts = detected.float().mean().item()
        else:
            recall_pts = float("nan")

        # IoU stayed : chevauchement entre pred >= stayed_threshold et target == 255
        pred_stayed  = (pred >= stayed_threshold)
        inter = (pred_stayed & mask_stayed).sum().item()
        union = (pred_stayed | mask_stayed).sum().item()
        iou_stayed = inter / union if union > 0 else float("nan")

    return {
        "mse_bg":     mse_bg,
        "mse_events": mse_events,
        "recall_pts": recall_pts,
        "iou_stayed": iou_stayed,
    }


# =============================================================================
# Boucle d'entraînement
# =============================================================================

def train(
    data_root: str,
    img_size: tuple = (512, 512),
    batch_size: int = 4,
    epochs: int = 20,
    lr: float = 1e-3,
    base_ch: int = 32,
    num_workers: int = 0,
    save_path: str = "crowd_tracking_net.pth",
) -> None:
    """
    Entraîne CrowdTrackingNet sur le dataset de tracking.

    Pipeline par batch :
        1. Concaténer Frame A et Frame B → (B, 6, H, W)
        2. Générer les cartes cibles à la volée via make_target_map
        3. Passe avant du modèle → prédiction (B, 1, H, W)
        4. Calcul de la perte pondérée
        5. Rétropropagation + mise à jour des poids Adam

    Args:
        data_root   : chemin vers data_tracking/ (issu de dataCleaning.py).
        img_size    : (H, W) fixe — nécessaire pour empiler les images en batch.
        batch_size  : paires par batch. Chaque paire = 2 images → mémoire ×2.
        epochs      : nombre de passes sur le dataset complet.
        lr          : learning rate initial (Adam).
        base_ch     : canaux de base du U-Net (16 / 32 / 64).
        num_workers : processus parallèles (0 = sûr sur Windows).
        save_path   : chemin de sauvegarde du meilleur modèle (.pth).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("test")
    print(f"Appareil : {device}")
    
    

    img_h, img_w = img_size

    print(f"Début de l'entraînement : {epochs} époques, batch_size={batch_size}, lr={lr}, base_ch={base_ch}")
    # ---- Chargement des données ----
    train_loader, _ = get_tracking_dataloader(
        data_root, split="train",
        batch_size=batch_size, img_size=img_size,
        augment=True, num_workers=num_workers,
    )
    print(f"  → {len(train_loader.dataset)} paires d'entraînement chargées.")
    val_loader, _ = get_tracking_dataloader(
        data_root, split="val",
        batch_size=batch_size, img_size=img_size,
        augment=False, num_workers=num_workers,
    )
    print(f"  → {len(val_loader.dataset)} paires de validation chargées.")

    # ---- Modèle, optimiseur, scheduler ----
    model = CrowdTrackingNet(base_ch=base_ch).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametres entrainables : {n_params:,}")

    optimizer = Adam(model.parameters(), lr=lr)
    # Réduit le lr ×0.5 toutes les 5 époques pour affiner progressivement
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoque {epoch}/{epochs}  —  LR : {scheduler.get_last_lr()[0]:.2e}")

        # ---- Phase entraînement ----
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, 1):
            print(f"  Epoch {epoch} — Batch {batch_idx}/{len(train_loader)}", end="\r")
            # Concaténation des deux frames sur la dimension canal
            # frame_A (B,3,H,W) + frame_B (B,3,H,W) → (B,6,H,W)
            x = torch.cat(
                [batch["frame_A"].to(device),
                 batch["frame_B"].to(device)], dim=1
            )

            # Cartes cibles générées à la volée depuis les annotations
            target = build_target_batch(batch, img_h, img_w, device)

            pred = model(x)
            loss = tracking_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Phase validation ----
        model.eval()
        val_loss = 0.0
        agg = {"mse_bg": 0.0, "mse_events": 0.0, "recall_pts": 0.0, "iou_stayed": 0.0}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = torch.cat(
                    [batch["frame_A"].to(device),
                     batch["frame_B"].to(device)], dim=1
                )
                target = build_target_batch(batch, img_h, img_w, device)
                pred = model(x)
                val_loss += tracking_loss(pred, target).item()
                m = compute_metrics(pred, target)
                for k in agg:
                    v = m[k]
                    if not (v != v):  # ignore NaN
                        agg[k] += v
                        if k == "mse_bg":
                            n_val += 1
        val_loss /= len(val_loader)
        if n_val > 0:
            for k in agg:
                agg[k] /= n_val

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoque {epoch:>3}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"mse_bg={agg['mse_bg']:.4f}  mse_ev={agg['mse_events']:.2f}  "
            f"recall_pts={agg['recall_pts']:.3f}  iou_stayed={agg['iou_stayed']:.3f}  "
            f"lr={current_lr:.2e}"
        )

        # Sauvegarde du meilleur modèle (critère : val_loss minimale)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Sauvegarde : {save_path}  (val_loss={val_loss:.4f})")

    print(f"\nEntraînement termine. Meilleure val_loss : {best_val_loss:.4f}")


# =============================================================================
# Smoke test
# =============================================================================

def smoke_test(data_root: str, img_size: tuple = (512, 512)) -> None:
    """
    Vérifie que le modèle tourne correctement sur un batch réel.

    Contrôles effectués :
        - Dimensions d'entrée et de sortie cohérentes
        - Valeurs dans la target map ({-1, 0, 1, 255})
        - Absence d'erreurs numériques (NaN, Inf)
        - Calcul de la perte sans crash

    Args:
        data_root : chemin vers data_tracking/
        img_size  : (H, W) utilisé pour le test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_h, img_w = img_size

    print("\n--- Smoke test CrowdTrackingNet ---")
    print(f"  Appareil : {device}  |  img_size : {img_size}")

    loader, _ = get_tracking_dataloader(
        data_root, split="train", batch_size=2,
        img_size=img_size, augment=False, num_workers=0,
    )
    batch = next(iter(loader))

    # ---- Entrée ----
    x = torch.cat([batch["frame_A"], batch["frame_B"]], dim=1).to(device)
    print(f"  Input  : {tuple(x.shape)}  (B, 6, H, W)")

    # ---- Target ----
    target = build_target_batch(batch, img_h, img_w, device)
    print(f"  Target : {tuple(target.shape)}  (B, 1, H, W)")

    # Valeurs présentes dans la carte cible (doit contenir certains {-1, 1, 255})
    unique = sorted({round(v, 0) for v in torch.unique(target).tolist()})
    print(f"  Valeurs cibles : {unique}")

    # ---- Passe avant ----
    model = CrowdTrackingNet(base_ch=32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x)

    print(f"  Output : {tuple(pred.shape)}  (B, 1, H, W)")
    print(f"  Plage sortie : [{pred.min().item():.3f},  {pred.max().item():.3f}]")

    # Vérification absence de NaN / Inf
    assert not torch.isnan(pred).any(), "NaN détecté dans la sortie !"
    assert not torch.isinf(pred).any(), "Inf détecté dans la sortie !"

    loss = tracking_loss(pred, target)
    print(f"  Perte initiale : {loss.item():.4f}")
    print("  Smoke test OK")


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    TRACKING_ROOT = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "crowd_tracking_net.pth")

    print(f"data/ : {TRACKING_ROOT}")

    if not os.path.isdir(TRACKING_ROOT):
        print("[ERREUR] data/ introuvable.")
        print("  -> Vérifiez le chemin vers le dataset JHU-CROWD++ original.")
    else:
        # smoke_test(TRACKING_ROOT, img_size=(512, 512))

        train(
            data_root  = TRACKING_ROOT,
            img_size   = (512, 512),
            batch_size = 4,
            epochs     = 20,
            lr         = 1e-3,
            base_ch    = 32,
            num_workers= 0,
            save_path  = SAVE_PATH,
        )
