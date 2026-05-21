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

Architecture choisie : U-Net Siamois (encodeur partagé + décodeur fusionné)
    • Entrée  : Frame A (B, 3, H, W) et Frame B (B, 3, H, W) séparées
    • Sortie  : carte de tracking (B, 1, H, W)

    Différence avec modelcopy.py (U-Net classique 6ch) :
        Ici chaque frame passe dans le MÊME encodeur (poids partagés).
        Les features sont fusionnées par concaténation au bottleneck et
        dans chaque skip connection.  Le décodeur traite donc des features
        qui représentent explicitement les deux frames comparées.

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
# Architecture principale : SiameseCrowdTrackingNet
# =============================================================================

class CrowdTrackingNet(nn.Module):
    """
    U-Net Siamois pour le tracking de foule.

    Principe : Frame A et Frame B passent chacune dans le MÊME encodeur
    (poids partagés).  Les features des deux branches sont concaténées
    au bottleneck ET dans chaque skip connection, puis le décodeur
    reconstruit la carte de tracking à partir de cette fusion.

    Avantage vs U-Net 6ch (modelcopy) :
        L'encodeur partagé apprend des features indépendantes de quelle
        frame il traite.  La comparaison A↔B se fait explicitement par
        concaténation, au lieu d'être implicite dans les 6 canaux d'entrée.

    Architecture (base_ch=32) :

        Frame A (B,3,H,W) ──► enc1(32)──► enc2(64)──► enc3(128)──► enc4(256)
                                  │            │             │            │
        Frame B (B,3,H,W) ──► enc1(32)──► enc2(64)──► enc3(128)──► enc4(256)
                             (poids partagés sur toute la colonne)
                                  │            │             │            │
                               skip1(64)   skip2(128)   skip3(256)   skip4(512)
                                                                         │
                                                              bottleneck(512→512)
                                                                         │
                              up4(256) + skip4(512) → dec4(768→256)
                              up3(128) + skip3(256) → dec3(384→128)
                              up2( 64) + skip2(128) → dec2(192→ 64)
                              up1( 32) + skip1( 64) → dec1( 96→ 32)
                                                       head(32→1)
                                                    Output (B,1,H,W)
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()

        # ---- Encodeur partagé (utilisé deux fois : pour A et pour B) -----
        self.enc1 = _conv_block(3,         base_ch)      # (B, 32,  H,    W)
        self.enc2 = _conv_block(base_ch,   base_ch * 2)  # (B, 64,  H/2,  W/2)
        self.enc3 = _conv_block(base_ch*2, base_ch * 4)  # (B, 128, H/4,  W/4)
        self.enc4 = _conv_block(base_ch*4, base_ch * 8)  # (B, 256, H/8,  W/8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- Bottleneck --------------------------------------------------
        # Entrée : concat(pool(e4_A), pool(e4_B)) = base_ch*16 canaux
        self.bottleneck = _conv_block(base_ch*16, base_ch*16)  # 512→512

        # ---- Décodeur ----------------------------------------------------
        # Les skips sont 2× plus larges car on concatène A et B.
        # Notation : up_out + skip = entrée dec
        #   dec4 : base_ch*8 + base_ch*16 = base_ch*24  (256+512=768)
        #   dec3 : base_ch*4 + base_ch* 8 = base_ch*12  (128+256=384)
        #   dec2 : base_ch*2 + base_ch* 4 = base_ch* 6  ( 64+128=192)
        #   dec1 : base_ch   + base_ch* 2 = base_ch* 3  ( 32+ 64= 96)
        self.up4  = nn.ConvTranspose2d(base_ch*16, base_ch*8,  kernel_size=2, stride=2)
        self.dec4 = _conv_block(base_ch*24, base_ch*8)

        self.up3  = nn.ConvTranspose2d(base_ch*8,  base_ch*4,  kernel_size=2, stride=2)
        self.dec3 = _conv_block(base_ch*12, base_ch*4)

        self.up2  = nn.ConvTranspose2d(base_ch*4,  base_ch*2,  kernel_size=2, stride=2)
        self.dec2 = _conv_block(base_ch*6,  base_ch*2)

        self.up1  = nn.ConvTranspose2d(base_ch*2,  base_ch,    kernel_size=2, stride=2)
        self.dec1 = _conv_block(base_ch*3,  base_ch)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def _encode(self, x: torch.Tensor):
        """Passe une frame (B,3,H,W) dans l'encodeur partagé."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def forward(self, frame_A: torch.Tensor, frame_B: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_A : (B, 3, H, W)
            frame_B : (B, 3, H, W)

        Returns:
            Tensor (B, 1, H, W) — carte de tracking prédite.
        """
        # ---- Encodage séparé des deux frames (poids partagés) ----
        e1_A, e2_A, e3_A, e4_A = self._encode(frame_A)
        e1_B, e2_B, e3_B, e4_B = self._encode(frame_B)

        # ---- Fusion au bottleneck ----
        b = self.bottleneck(
            torch.cat([self.pool(e4_A), self.pool(e4_B)], dim=1)
        )  # (B, 512, H/16, W/16)

        # ---- Skip connections doublées (concat A+B à chaque niveau) ----
        skip4 = torch.cat([e4_A, e4_B], dim=1)  # (B, 512, H/8,  W/8)
        skip3 = torch.cat([e3_A, e3_B], dim=1)  # (B, 256, H/4,  W/4)
        skip2 = torch.cat([e2_A, e2_B], dim=1)  # (B, 128, H/2,  W/2)
        skip1 = torch.cat([e1_A, e1_B], dim=1)  # (B,  64, H,    W)

        # ---- Décodeur ----
        d4 = self.dec4(torch.cat([self.up4(b),  skip4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), skip3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skip2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), skip1], dim=1))

        return self.head(d1)  # (B, 1, H, W)


# =============================================================================
# Fonction de perte
# =============================================================================

def tracking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_bg: float = 1.0,
    w_stayed: float = 5.0,
    w_pts: float = 200.0,
) -> torch.Tensor:
    """
    MSE pondérée avec poids séparés par type d'événement.

    Pourquoi des poids différents ?
        • fond (0)       : majorité absolue des pixels → poids faible (1)
        • stayed (255)   : traits de plusieurs pixels chacun → présence non négligeable
                           mais valeur 255 crée déjà un fort signal MSE → poids modéré (5)
        • pts (±1)       : un seul pixel par tête entered/left → extrêmement rare
                           ET valeur faible (±1) → signal MSE minuscule sans pondération
                           → poids très élevé (200) pour forcer le réseau à les apprendre

    Args:
        pred     : sortie du modèle (B, 1, H, W).
        target   : carte cible      (B, 1, H, W).
        w_bg     : poids des pixels de fond (valeur 0).
        w_stayed : poids des pixels de trajectoires stayed (valeur 255).
        w_pts    : poids des pixels entered (+1) et left (-1).

    Returns:
        Tensor scalaire différentiable.
    """
    mask_stayed = (target == 255).float()
    mask_pts    = (target.abs() == 1).float()
    mask_bg     = (target == 0).float()

    weights = mask_bg * w_bg + mask_stayed * w_stayed + mask_pts * w_pts

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
            frame_A = batch["frame_A"].to(device)  # (B, 3, H, W)
            frame_B = batch["frame_B"].to(device)  # (B, 3, H, W)

            target = build_target_batch(batch, img_h, img_w, device)

            pred = model(frame_A, frame_B)
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
                frame_A = batch["frame_A"].to(device)
                frame_B = batch["frame_B"].to(device)
                target = build_target_batch(batch, img_h, img_w, device)
                pred = model(frame_A, frame_B)
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
    frame_A = batch["frame_A"].to(device)
    frame_B = batch["frame_B"].to(device)
    print(f"  Input  : frame_A {tuple(frame_A.shape)}  frame_B {tuple(frame_B.shape)}")

    # ---- Target ----
    target = build_target_batch(batch, img_h, img_w, device)
    print(f"  Target : {tuple(target.shape)}  (B, 1, H, W)")

    unique = sorted({round(v, 0) for v in torch.unique(target).tolist()})
    print(f"  Valeurs cibles : {unique}")

    # ---- Passe avant ----
    model = CrowdTrackingNet(base_ch=32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(frame_A, frame_B)

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
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "crowd_tracking_net_siamese.pth")

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
