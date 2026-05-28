"""
modelcopyV20.py — CrowdTrackingNet  (tâche simplifiée : stayed uniquement)
===========================================================================

Changements majeurs par rapport aux versions précédentes (V15–V18) :

  1. Suppression totale de "entered" et "left".
       Raison : le signal entered/left est trop rare et bruité dans les scènes
       denses pour être utile — il dominait en faux positifs sans apporter de
       valeur mesurable.  La tâche se réduit à :
           "Où se trouvait chaque tête dans Frame A, et où est-elle dans Frame B ?"

  2. Remplacement du trait épais (cv2.line, épaisseur 3) par deux points
     gaussiens serrés (σ = SIGMA_STAYED = 3 px) aux deux extrémités de chaque
     paire stayed (position Frame A et position Frame B).
       Raison : le trait supervisait toute la TRAJECTOIRE, ce qui encourageait
       le modèle à prédire de larges "blobs" le long du chemin.  Deux points
       serrés focalisent la supervision sur les positions exactes des têtes.

  3. Simplification de la loss : uniquement w_bg × MSE_fond + w_stayed × MSE_stayed.
       mask_stayed couvre toute la gaussienne (target > 0.05), pas seulement le
       pic (target >= 1.5), ce qui donne un gradient directionnel vers le centre.

Architecture : U-Net (6 canaux d'entrée → 1 canal de sortie), inchangée.
"""

import os
import random
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from dataGathering import get_tracking_dataloader

STAYED_VAL   = 2.0   # valeur cible au pic de la gaussienne stayed
SIGMA_STAYED = 3     # σ serré (px) — rayon effectif ≈ 3σ = 9 px par point


def _draw_gaussian(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    amplitude: float,
    sigma: float = SIGMA_STAYED,
) -> None:
    """
    Dessine une gaussienne 2D centrée sur (cx, cy) sur canvas (en place).
    Région calculée : ±3σ.  Utilise np.maximum pour éviter l'annulation
    entre têtes proches de même classe.
    """
    h, w = canvas.shape
    r = max(1, int(3 * sigma))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    stayed_mask = canvas[y0:y1, x0:x1] >= 1.5
    if amplitude > 0:
        result = np.maximum(canvas[y0:y1, x0:x1], patch)
    else:
        result = np.minimum(canvas[y0:y1, x0:x1], patch)
    result[stayed_mask] = canvas[y0:y1, x0:x1][stayed_mask]
    canvas[y0:y1, x0:x1] = result


# =============================================================================
# Génération de la carte cible
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

    Valeurs encodées (V19 — stayed uniquement) :
        0          → arrière-plan
        0–STAYED_VAL → gaussienne 2D centrée sur la position d'une tête stayed
                       (pic = STAYED_VAL = 2.0, σ = SIGMA_STAYED = 3 px)

    Pour chaque paire stayed, DEUX gaussiennes sont dessinées :
        • position de la tête dans Frame B  (x_B, y_B)
        • position de la tête dans Frame A projetée en coords Frame B  (x_B+dx, y_B+dy)

    Avantage vs trait épais : le modèle est guidé vers les POINTS exacts des
    têtes, pas vers toute la trajectoire — évite les blobs le long du trajet.
    """
    target = np.zeros((img_h, img_w), dtype=np.float32)

    if links:
        iA0, iB0 = links[0][0], links[0][1]
        dx = gt_A[iA0]["x"] - gt_B[iB0]["x"]
        dy = gt_A[iA0]["y"] - gt_B[iB0]["y"]
    else:
        dx, dy = 0, 0

    for pair in links:
        iB = int(pair[1])
        x_B, y_B = gt_B[iB]["x"], gt_B[iB]["y"]

        # Point 1 : position de la tête dans Frame B
        _draw_gaussian(target, x_B, y_B, amplitude=STAYED_VAL, sigma=SIGMA_STAYED)

        # Point 2 : position de la tête dans Frame A (coordonnées Frame B)
        x_A = x_B + dx
        y_A = y_B + dy
        if 0 <= x_A < img_w and 0 <= y_A < img_h:
            _draw_gaussian(target, x_A, y_A, amplitude=STAYED_VAL, sigma=SIGMA_STAYED)

    return torch.from_numpy(target).unsqueeze(0)


def build_target_batch(
    batch: dict,
    img_h: int,
    img_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Génère les cartes cibles pour tout un batch (B, 1, H, W)."""

    def boxes_to_anns(boxes: torch.Tensor) -> list[dict]:
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
        links = batch["links"][i].tolist()
        targets.append(make_target_map(gt_A, gt_B, links, img_h, img_w))

    return torch.stack(targets).to(device)


# =============================================================================
# Blocs de construction du réseau
# =============================================================================

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bloc double convolution Conv→BN→GELU→Conv→BN→GELU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


# =============================================================================
# Architecture : CrowdTrackingNet (U-Net)
# =============================================================================

class CrowdTrackingNet(nn.Module):
    """
    U-Net encodeur-décodeur pour le tracking stayed entre deux frames.

        Input  (B, 6, H, W)  ← Frame A + Frame B concaténées
        Output (B, 1, H, W)  ← heatmap stayed (pic ≈ STAYED_VAL aux positions de tête)
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.enc1 = _conv_block(6,         base_ch)
        self.enc2 = _conv_block(base_ch,   base_ch * 2)
        self.enc3 = _conv_block(base_ch*2, base_ch * 4)
        self.enc4 = _conv_block(base_ch*4, base_ch * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _conv_block(base_ch*8, base_ch*16)

        self.up4  = nn.ConvTranspose2d(base_ch*16, base_ch*8,  kernel_size=2, stride=2)
        self.dec4 = _conv_block(base_ch*16, base_ch*8)

        self.up3  = nn.ConvTranspose2d(base_ch*8,  base_ch*4,  kernel_size=2, stride=2)
        self.dec3 = _conv_block(base_ch*8,  base_ch*4)

        self.up2  = nn.ConvTranspose2d(base_ch*4,  base_ch*2,  kernel_size=2, stride=2)
        self.dec2 = _conv_block(base_ch*4,  base_ch*2)

        self.up1  = nn.ConvTranspose2d(base_ch*2,  base_ch,    kernel_size=2, stride=2)
        self.dec1 = _conv_block(base_ch*2,  base_ch)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# =============================================================================
# Fonction de perte
# =============================================================================

def tracking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_bg: float = 8.0,
    w_stayed: float = 8.0,
) -> torch.Tensor:
    """
    MSE pondérée par classe (fond vs stayed).

    mask_stayed couvre TOUTE la gaussienne (target > 0.05), pas seulement le pic.
    Cela donne un gradient directionnel vers le centre : le modèle apprend que
    la valeur doit croître vers (cx, cy) plutôt que d'être uniforme dans la zone.

    w_bg = w_stayed = 8.0 : les deux classes ont le même poids relatif par pixel,
    mais la zone stayed est bien plus petite que le fond, donc le gradient total
    est équilibré.
    """
    mask_stayed = (target > 0.05)       # gaussienne complète (pic + queues)
    mask_bg     = ~mask_stayed          # tout le reste est fond

    err_sq = (pred - target) ** 2

    loss_stayed = err_sq[mask_stayed].mean() if mask_stayed.any() else pred.new_tensor(0.0)
    loss_bg     = err_sq[mask_bg].mean()     if mask_bg.any()     else pred.new_tensor(0.0)

    return w_stayed * loss_stayed + w_bg * loss_bg


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    stayed_threshold: float = 1.5,
) -> dict:
    """
    Calcule les métriques stayed.

    Returns:
        mse_bg      — MSE sur les pixels de fond
        mse_stayed  — MSE sur les pixels stayed (gaussienne complète)
        iou_stayed  — IoU entre pred >= stayed_threshold et target > 0.05
    """
    with torch.no_grad():
        # Deux masques intentionnellement différents :
        # - full : gaussienne complète → pour mse_stayed (qualité du fit de forme)
        # - peak : pic uniquement      → pour iou_stayed (localisation des têtes)
        mask_stayed_full = (target > 0.05)
        mask_stayed_peak = (target >= 1.5)
        mask_bg          = ~mask_stayed_full

        mse_bg = ((pred[mask_bg] - target[mask_bg]) ** 2).mean().item() \
                 if mask_bg.any() else 0.0

        mse_stayed = ((pred[mask_stayed_full] - target[mask_stayed_full]) ** 2).mean().item() \
                     if mask_stayed_full.any() else 0.0

        pred_stayed = (pred >= stayed_threshold)
        tp    = (pred_stayed &  mask_stayed_peak).sum().item()
        fp    = (pred_stayed & ~mask_stayed_peak).sum().item()
        fn    = (~pred_stayed & mask_stayed_peak).sum().item()
        denom_iou  = tp + fp + fn
        iou_stayed       = tp / denom_iou        if denom_iou   > 0 else float("nan")
        precision_stayed = tp / (tp + fp)        if (tp + fp)   > 0 else float("nan")
        recall_stayed    = tp / (tp + fn)        if (tp + fn)   > 0 else float("nan")

    return {
        "mse_bg":           mse_bg,
        "mse_stayed":       mse_stayed,
        "iou_stayed":       iou_stayed,
        "precision_stayed": precision_stayed,
        "recall_stayed":    recall_stayed,
    }


# =============================================================================
# Visualisation
# =============================================================================

def _colorize_map(arr: np.ndarray, is_prediction: bool = False) -> np.ndarray:
    """
    Convertit une carte (H, W) float en image BGR (H, W, 3) uint8.
        Stayed → blanc  (255, 255, 255)
        Fond   → noir   (  0,   0,   0)

    Pour la target, seuil bas (0.5) pour visualiser l'étendue de la gaussienne.
    Pour la prédiction, seuil haut (1.5) pour n'afficher que les pics réels.
    """
    h, w = arr.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    if is_prediction:
        stayed_mask = arr >= 1.5
    else:
        stayed_mask = arr > 0.5   # montre la gaussienne cible jusqu'à 25% du pic
    bgr[stayed_mask] = (255, 255, 255)
    return bgr


def visualize_predictions(
    model_path: str,
    data_root: str,
    output_dir: str,
    img_size: tuple = (512, 512),
    n_samples: int = 8,
    base_ch: int = 32,
    split: str = "test",
    indices: list[int] | None = None,
) -> None:
    """
    Charge le modèle et sauvegarde des images de débogage côte à côte :
        Frame A | Frame B | Target | Pred (seuil) | Pred brute (heatmap)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_h, img_w = img_size

    model = CrowdTrackingNet(base_ch=base_ch).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Modèle chargé depuis : {model_path}")

    _, dataset = get_tracking_dataloader(
        data_root, split=split, batch_size=1,
        img_size=img_size, augment=False, num_workers=0,
    )

    if indices is not None:
        selected = indices
    else:
        n = min(n_samples, len(dataset))
        selected = random.sample(range(len(dataset)), n)

    print(f"Visualisation de {len(selected)} images (split={split}) : indices {selected}")

    for sample_num, ds_idx in enumerate(selected):
        sample = dataset[ds_idx]
        batch = {
            "frame_A": sample["frame_A"].unsqueeze(0),
            "frame_B": sample["frame_B"].unsqueeze(0),
            "boxes_A": [sample["boxes_A"]],
            "boxes_B": [sample["boxes_B"]],
            "links":   [sample["links"]],
        }

        x      = torch.cat([batch["frame_A"].to(device), batch["frame_B"].to(device)], dim=1)
        target = build_target_batch(batch, img_h, img_w, device)

        with torch.no_grad():
            pred = model(x)

        frame_a  = batch["frame_A"][0].permute(1, 2, 0).numpy()
        frame_b  = batch["frame_B"][0].permute(1, 2, 0).numpy()
        tgt_map  = target[0, 0].cpu().numpy()
        pred_map = pred[0, 0].cpu().numpy()

        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        def to_bgr(arr):
            arr = np.clip(arr * _std + _mean, 0.0, 1.0)
            return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        fa_bgr = to_bgr(frame_a)
        fb_bgr = to_bgr(frame_b)

        tgt_color   = _colorize_map(tgt_map,  is_prediction=False)
        pred_thresh = _colorize_map(pred_map, is_prediction=True)

        p_min, p_max = pred_map.min(), pred_map.max()
        pred_norm = (pred_map - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(pred_map)
        pred_heat = cv2.applyColorMap((pred_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        def add_label(img, text):
            out = img.copy()
            cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return out

        panels = [
            add_label(fa_bgr,      "Frame A"),
            add_label(fb_bgr,      "Frame B"),
            add_label(tgt_color,   "Target"),
            add_label(pred_thresh, "Pred (seuil)"),
            add_label(pred_heat,   f"Pred brute [{p_min:.2f},{p_max:.2f}]"),
        ]
        combined = np.concatenate(panels, axis=1)

        # Comptages GT
        gt_A      = sample.get("count_A",      "?")
        gt_B      = sample.get("count_B",      "?")
        gt_stayed = sample.get("count_stayed", "?")

        # Comptage prédit : pics au-dessus du seuil stayed (1 pic ≈ 1 tête)
        def count_peaks(arr, threshold, min_dist=10):
            if (arr > threshold).sum() == 0:
                return 0
            kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_dist, min_dist))
            dilated = cv2.dilate(arr, kernel)
            peaks   = (arr == dilated) & (arr > threshold)
            return int(peaks.sum())

        # Avec 2 points par paire stayed, on divise par 2 pour estimer le nb de têtes
        pr_peaks  = count_peaks(pred_map, threshold=1.5, min_dist=10)
        pr_stayed = pr_peaks // 2   # 2 pics par paire stayed

        header_h = 56
        header   = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
        gt_line   = (f"GT :   Frame A={gt_A} pers.  Frame B={gt_B} pers.  "
                     f"[ stayed={gt_stayed} ]")
        pred_line = (f"Pred : stayed~{pr_stayed}  (pics bruts={pr_peaks}, ÷2 car 2 pts/paire)")
        cv2.putText(header, gt_line,   (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(header, pred_line, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255),   1)
        combined = np.vstack([header, combined])

        m = compute_metrics(pred, target)
        stats = (f"iou={m['iou_stayed']:.3f}  prec={m['precision_stayed']:.3f}  rec={m['recall_stayed']:.3f}  "
                 f"mse_bg={m['mse_bg']:.4f}  mse_stayed={m['mse_stayed']:.3f}")
        cv2.putText(combined, stats, (8, combined.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        out_path = os.path.join(output_dir, f"sample_{ds_idx:04d}.png")
        cv2.imwrite(out_path, combined)
        print(f"  [{sample_num+1}/{len(selected)}] idx={ds_idx} → {out_path}  iou={m['iou_stayed']:.3f}  prec={m['precision_stayed']:.3f}  rec={m['recall_stayed']:.3f}")

    print(f"\nVisualisations sauvegardées dans : {output_dir}")


# =============================================================================
# Courbes d'entraînement
# =============================================================================

def _plot_training_history(history: dict, save_path: str, best_epoch: int) -> None:
    """Génère et sauvegarde un graphe 2×2 des courbes d'entraînement."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Courbes d'entraînement", fontsize=14, fontweight="bold")

    def _vline(ax):
        if best_epoch > 0:
            ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1,
                       label=f"meilleur (ép. {best_epoch})")

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train loss", color="steelblue")
    ax.plot(epochs, history["val_loss"],   label="val loss",   color="darkorange")
    _vline(ax)
    ax.set_title("Loss")
    ax.set_xlabel("Époque")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["iou_stayed"],       label="IoU",       color="purple")
    ax.plot(epochs, history["precision_stayed"], label="Précision",  color="steelblue")
    ax.plot(epochs, history["recall_stayed"],    label="Recall",     color="green")
    _vline(ax)
    ax.set_title("IoU / Précision / Recall stayed  (higher = better)")
    ax.set_xlabel("Époque")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["mse_bg"],     label="MSE fond",    color="gray")
    ax.plot(epochs, history["mse_stayed"], label="MSE stayed",  color="crimson")
    _vline(ax)
    ax.set_title("MSE par région")
    ax.set_xlabel("Époque")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], color="teal")
    _vline(ax)
    ax.set_title("Learning rate")
    ax.set_xlabel("Époque")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"  Courbes sauvegardées : {save_path}")
    plt.show()
    plt.close(fig)


# =============================================================================
# Boucle d'entraînement
# =============================================================================

def train(
    data_root: str,
    img_size: tuple = (512, 512),
    batch_size: int = 4,
    epochs: int = 50,
    lr: float = 1e-3,
    base_ch: int = 32,
    num_workers: int = 0,
    save_path: str = "crowd_tracking_net.pth",
) -> None:
    """Entraîne CrowdTrackingNet (tâche stayed uniquement)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil : {device}")
    img_h, img_w = img_size

    print(f"Début entraînement : {epochs} époques  batch={batch_size}  lr={lr}  base_ch={base_ch}")

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

    model = CrowdTrackingNet(base_ch=base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {n_params:,}")

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_epoch    = 0

    history = {
        "train_loss": [], "val_loss": [],
        "iou_stayed": [], "precision_stayed": [], "recall_stayed": [],
        "mse_bg": [], "mse_stayed": [], "lr": [],
    }

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoque {epoch}/{epochs}  —  LR : {current_lr:.2e}")

        # ---- Entraînement ----
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, 1):
            print(f"  Epoch {epoch} — Batch {batch_idx}/{len(train_loader)}", end="\r")
            x = torch.cat([batch["frame_A"].to(device), batch["frame_B"].to(device)], dim=1)
            target = build_target_batch(batch, img_h, img_w, device)
            pred   = model(x)
            loss   = tracking_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        agg = {"mse_bg": 0.0, "mse_stayed": 0.0, "iou_stayed": 0.0,
               "precision_stayed": 0.0, "recall_stayed": 0.0}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = torch.cat([batch["frame_A"].to(device), batch["frame_B"].to(device)], dim=1)
                target = build_target_batch(batch, img_h, img_w, device)
                pred   = model(x)
                val_loss += tracking_loss(pred, target).item()
                m = compute_metrics(pred, target)
                for k in agg:
                    v = m[k]
                    if v == v:   # ignore NaN
                        agg[k] += v
                        if k == "mse_bg":
                            n_val += 1
        val_loss /= len(val_loader)
        if n_val > 0:
            for k in agg:
                agg[k] /= n_val

        iou  = agg["iou_stayed"]
        prec = agg["precision_stayed"]
        rec  = agg["recall_stayed"]
        iou_valid  = iou  if iou  == iou  else 0.0
        prec_valid = prec if prec == prec else 0.0
        rec_valid  = rec  if rec  == rec  else 0.0

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoque {epoch:>3}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"mse_bg={agg['mse_bg']:.4f}  mse_stayed={agg['mse_stayed']:.4f}  "
            f"iou={iou_valid:.3f}  prec={prec_valid:.3f}  rec={rec_valid:.3f}  lr={current_lr:.2e}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["iou_stayed"].append(iou_valid)
        history["precision_stayed"].append(prec_valid)
        history["recall_stayed"].append(rec_valid)
        history["mse_bg"].append(agg["mse_bg"])
        history["mse_stayed"].append(agg["mse_stayed"])
        history["lr"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  -> Sauvegarde : {save_path}  (val_loss={val_loss:.4f}  iou={iou_valid:.3f}  prec={prec_valid:.3f}  rec={rec_valid:.3f})")

    print(f"\nEntraînement terminé. Meilleur : ép.{best_epoch}  val_loss={best_val_loss:.4f}")
    plot_path = save_path.replace(".pth", "_courbes.png")
    _plot_training_history(history, plot_path, best_epoch)


# =============================================================================
# Smoke test
# =============================================================================

def smoke_test(data_root: str, img_size: tuple = (512, 512)) -> None:
    """Vérifie que le modèle tourne correctement sur un batch réel."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_h, img_w = img_size
    print("\n--- Smoke test CrowdTrackingNet (V19) ---")
    print(f"  Appareil : {device}  |  img_size : {img_size}")

    loader, _ = get_tracking_dataloader(
        data_root, split="train", batch_size=2,
        img_size=img_size, augment=False, num_workers=0,
    )
    batch = next(iter(loader))
    x = torch.cat([batch["frame_A"], batch["frame_B"]], dim=1).to(device)
    target = build_target_batch(batch, img_h, img_w, device)

    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Target : {tuple(target.shape)}  plage=[{target.min():.2f}, {target.max():.2f}]")

    model = CrowdTrackingNet(base_ch=32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x)

    assert not torch.isnan(pred).any(), "NaN dans la sortie !"
    assert not torch.isinf(pred).any(), "Inf dans la sortie !"

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
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "crowd_tracking_net20.pth")
    VIZ_DIR   = os.path.join(os.path.dirname(__file__), "..", "visualizations/20")

    # "train"     → entraîne et sauvegarde le meilleur checkpoint
    # "visualize" → charge le checkpoint et génère les images de débogage
    MODE = "train"

    print(f"data/ : {TRACKING_ROOT}")

    if not os.path.isdir(TRACKING_ROOT):
        print("[ERREUR] data/ introuvable.")
        print("  -> Vérifiez le chemin vers le dataset JHU-CROWD++ original.")
    elif MODE == "train":
        train(
            data_root   = TRACKING_ROOT,
            img_size    = (512, 512),
            batch_size  = 16,
            epochs      = 50,
            lr          = 1e-3,
            base_ch     = 32,
            num_workers = 12,
            save_path   = SAVE_PATH,
        )
    elif MODE == "visualize":
        if not os.path.isfile(SAVE_PATH):
            print(f"[ERREUR] Checkpoint introuvable : {SAVE_PATH}")
            print("  -> Entraînez d'abord le modèle avec MODE='train'.")
        else:
            visualize_predictions(
                model_path = SAVE_PATH,
                data_root  = TRACKING_ROOT,
                output_dir = VIZ_DIR,
                img_size   = (512, 512),
                n_samples  = 8,
                base_ch    = 32,
                split      = "test",
            )
