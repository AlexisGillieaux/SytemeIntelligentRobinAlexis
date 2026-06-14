"""
modelcopyV49.py — CrowdTrackingNet  (σ=15, 1024×1024, 100 époques, MSE pondérée + offsets)
===========================================================================

Nouveautés V49 (révisé — base : V48) :

  L'expérience « focal loss » initiale de V49 a fortement DÉGRADÉ les
  performances (sorties sigmoïdes sous-confiantes plafonnant à ~0.8, très
  peu de détections au seuil, métriques Frame B effondrées vs V44-V48).
  Cause probable : avec σ=15 (grandes gaussiennes), le schéma focal pénalisé
  (un seul pixel-centre positif + suppression en (1−cible)^4 sur une large
  zone) pousse vers des heatmaps diffuses et faibles.

  → Retour au régime ÉPROUVÉ de V44-V48 :
      - MSE pondérée par zone (peak / tail / fond) au lieu de la focal loss.
      - Cibles gaussiennes d'amplitude 2.0, sorties BRUTES (pas de sigmoïde).
      - Seuils d'évaluation 1.5, min_dist=30 (comme V44-V48).
      - Poids : w_b_peak=w_a_peak=40, w_tail=4, w_bg=8, w_off=20.

  Conservé de V48 : tête d'offsets (canaux 2-3) pour l'association A↔B,
  supervision L1 masquée, flèches vertes dans les visus, warp géométrique,
  σ=15, LR warmup 5 ép.

Architecture : U-Net (6 canaux d'entrée → 4 canaux de sortie ;
canaux 0-1 heatmaps brutes, canaux 2-3 offsets).
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, os.path.dirname(__file__))
from dataGathering import get_tracking_dataloader

STAYED_VAL_B = 2.0   # amplitude gaussiennes canal 0 — Frame B (positions actuelles)
STAYED_VAL_A = 2.0   # amplitude gaussiennes canal 1 — Frame A (positions précédentes)
SIGMA_STAYED = 15    # σ (px) — rayon effectif ≈ 3σ = 45 px par point
OFFSET_NORM  = 32.0  # normalisation des offsets (px) → cibles ~[-1.5, 1.5]
R_OFFSET     = 8     # rayon (px) du disque de supervision des offsets


def _draw_gaussian(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    amplitude: float,
    sigma: float = SIGMA_STAYED,
) -> None:
    """
    Dessine une gaussienne 2D positive centrée sur (cx, cy) sur canvas (en place).
    Utilise np.maximum pour préserver le pic le plus fort sur les chevauchements.
    """
    h, w = canvas.shape
    r = max(1, int(3 * sigma))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], patch)


def _fill_disk(
    canvas: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    value: float,
    radius: int = 8,
) -> None:
    """
    Écrit ``value`` sur un disque de rayon ``radius`` centré en (cx, cy) et
    met le masque à 1 au même endroit (en place).  Sert aux cibles d'offsets :
    chaque tête écrit SON offset autour de son pic Frame B.
    """
    h, w = canvas.shape
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    canvas[y0:y1, x0:x1][disk] = value
    mask[y0:y1, x0:x1][disk] = 1.0


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
    Génère la carte cible 5 canaux pour une paire (Frame A, Frame B).

    Encodage V49 — heatmaps (amplitude 2.0) + offsets CenterTrack :
        Canal 0 : +2.0 aux positions ACTUELLES (Frame B)
        Canal 1 : +2.0 aux positions PRÉCÉDENTES (Frame A)
        Canal 2 : Δx/OFFSET_NORM — décalage horizontal B→A de chaque tête,
                  écrit sur un disque de rayon R_OFFSET autour de son pic B
        Canal 3 : Δy/OFFSET_NORM — idem, vertical
        Canal 4 : masque de supervision des offsets (1 sur les disques)

    Sortie : (5, H, W) float32.  Le modèle ne prédit que les canaux 0-3 ;
    le canal 4 ne sert qu'à masquer la perte L1 des offsets.
    """
    canvas_B = np.zeros((img_h, img_w), dtype=np.float32)
    canvas_A = np.zeros((img_h, img_w), dtype=np.float32)
    off_x    = np.zeros((img_h, img_w), dtype=np.float32)
    off_y    = np.zeros((img_h, img_w), dtype=np.float32)
    mask_off = np.zeros((img_h, img_w), dtype=np.float32)

    for pair in links:
        iA, iB = int(pair[0]), int(pair[1])
        x_B, y_B = gt_B[iB]["x"], gt_B[iB]["y"]

        # Canal 0 — Frame B : position actuelle
        _draw_gaussian(canvas_B, x_B, y_B, amplitude=STAYED_VAL_B, sigma=SIGMA_STAYED)

        # Canal 1 — Frame A : position précédente RÉELLE de cette tête,
        # lue via le lien (et non reconstruite par un offset global) —
        # exact même quand le décalage varie localement (warp géométrique).
        x_A, y_A = gt_A[iA]["x"], gt_A[iA]["y"]
        if 0 <= x_A < img_w and 0 <= y_A < img_h:
            _draw_gaussian(canvas_A, x_A, y_A, amplitude=STAYED_VAL_A, sigma=SIGMA_STAYED)

        # Canaux 2-4 — offset B→A de CETTE tête, sur un disque autour du pic B
        _fill_disk(off_x, mask_off, x_B, y_B, (x_A - x_B) / OFFSET_NORM, R_OFFSET)
        _fill_disk(off_y, mask_off, x_B, y_B, (y_A - y_B) / OFFSET_NORM, R_OFFSET)

    return torch.from_numpy(
        np.stack([canvas_B, canvas_A, off_x, off_y, mask_off], axis=0)
    )  # (5, H, W)


def build_target_batch(
    batch: dict,
    img_h: int,
    img_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Génère les cartes cibles pour tout un batch → (B, 5, H, W)."""

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
        targets.append(make_target_map(gt_A, gt_B, links, img_h, img_w))  # (5, H, W)

    return torch.stack(targets).to(device)  # (B, 5, H, W)


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
# Architecture : CrowdTrackingNet (U-Net, 2 canaux de sortie)
# =============================================================================

class CrowdTrackingNet(nn.Module):
    """
    U-Net encodeur-décodeur pour le tracking stayed entre deux frames.

        Input  (B, 6, H, W)  ← Frame A + Frame B concaténées
        Output (B, 4, H, W)  ← canal 0   : heatmap Frame B
                                 canal 1   : heatmap Frame A
                                 canaux 2-3 : offsets (Δx, Δy) B→A, normalisés
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

        # Têtes séparées : chaque canal a sa propre représentation finale
        self.head_B = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(base_ch // 2, 1, kernel_size=1),
        )
        self.head_A = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(base_ch // 2, 1, kernel_size=1),
        )
        # Tête d'offsets (Δx, Δy) façon CenterTrack — 2 canaux de régression
        self.head_off = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(base_ch // 2, 2, kernel_size=1),
        )

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
        # Canaux 0-1 : heatmaps brutes (MSE). Canaux 2-3 : offsets (L1).
        return torch.cat([self.head_B(d1), self.head_A(d1), self.head_off(d1)], dim=1)


# =============================================================================
# Fonction de perte
# =============================================================================

def tracking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_b_peak: float = 40.0,
    w_a_peak: float = 40.0,
    w_tail:   float = 4.0,
    w_bg:     float = 8.0,
    w_off:    float = 20.0,
) -> torch.Tensor:
    """
    MSE pondérée par zone sur les 2 heatmaps + L1 masquée sur les offsets.

    Pour chaque canal ch ∈ {0=FrameB, 1=FrameA} :
        mask_peak : target[ch] ≥ 1.5   — centre gaussien (pic 2.0)
                    poids : w_b_peak (ch0) ou w_a_peak (ch1)
        mask_tail : 0.05 < target[ch] < 1.5 — queue gaussienne, poids w_tail
        mask_bg   : target[ch] ≤ 0.05  — fond pur, poids w_bg

    Régime éprouvé V44-V48 (la focal loss testée en V49 dégradait les
    performances). Les 2 canaux sont supervisés indépendamment.

    Offsets (canaux 2-3) : L1 masquée par le canal 4 de la cible
        (disques autour des pics B), pondérée par w_off.
    """
    total = pred.new_tensor(0.0)
    for ch, w_peak in enumerate([w_b_peak, w_a_peak]):
        p = pred[:, ch:ch+1]
        t = target[:, ch:ch+1]
        mask_peak = (t >= 1.5)
        mask_tail = (t > 0.05) & ~mask_peak
        mask_bg   = (t <= 0.05)
        err_sq    = (p - t) ** 2
        loss_peak = err_sq[mask_peak].mean() if mask_peak.any() else p.new_tensor(0.0)
        loss_tail = err_sq[mask_tail].mean() if mask_tail.any() else p.new_tensor(0.0)
        loss_bg   = err_sq[mask_bg].mean()   if mask_bg.any()   else p.new_tensor(0.0)
        total = total + w_peak * loss_peak + w_tail * loss_tail + w_bg * loss_bg

    # --- Terme offsets : L1 masquée sur les canaux 2-3 ---
    mask_off = target[:, 4:5]
    n_off    = mask_off.sum()
    if n_off > 0:
        off_err  = (pred[:, 2:4] - target[:, 2:4]).abs() * mask_off
        loss_off = off_err.sum() / (n_off * 2)
        total    = total + w_off * loss_off

    return total


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 1.5,
) -> dict:
    """
    Calcule les métriques pour les deux canaux.

    pred   : (B, 4, H, W) — canaux 0-1 heatmaps brutes, 2-3 offsets
    target : (B, 5, H, W) — canaux 0-1 heatmaps, 2-3 offsets, 4 masque
        canal 0 → Frame B (positions actuelles)
        canal 1 → Frame A (positions précédentes)

    Returns : mse_bg, mse_stayed,
              iou_b, precision_b, recall_b,
              iou_a, precision_a, recall_a,
              mae_off (erreur moyenne des offsets, en pixels)
    """
    with torch.no_grad():
        pred_B = pred[:, 0:1];    tgt_B = target[:, 0:1]
        pred_A = pred[:, 1:2];    tgt_A = target[:, 1:2]

        mask_peak_B = (tgt_B >= 1.5)
        mask_peak_A = (tgt_A >= 1.5)
        mask_bg_B   = (tgt_B <= 0.05)
        mask_bg_A   = (tgt_A <= 0.05)
        mask_stay_B = ~mask_bg_B
        mask_stay_A = ~mask_bg_A

        # MSE fond et zones actives (moyenne des deux canaux)
        mse_bg_b = ((pred_B[mask_bg_B] - tgt_B[mask_bg_B])**2).mean().item() \
                   if mask_bg_B.any() else 0.0
        mse_bg_a = ((pred_A[mask_bg_A] - tgt_A[mask_bg_A])**2).mean().item() \
                   if mask_bg_A.any() else 0.0
        mse_bg   = (mse_bg_b + mse_bg_a) / 2

        mse_stay_b = ((pred_B[mask_stay_B] - tgt_B[mask_stay_B])**2).mean().item() \
                     if mask_stay_B.any() else 0.0
        mse_stay_a = ((pred_A[mask_stay_A] - tgt_A[mask_stay_A])**2).mean().item() \
                     if mask_stay_A.any() else 0.0
        mse_stayed = (mse_stay_b + mse_stay_a) / 2

        def _iou_prec_rec(pred_ch, mask_peak):
            pred_bin = (pred_ch >= threshold)
            tp = (pred_bin &  mask_peak).sum().item()
            fp = (pred_bin & ~mask_peak).sum().item()
            fn = (~pred_bin & mask_peak).sum().item()
            iou  = tp / (tp+fp+fn) if (tp+fp+fn) > 0 else float("nan")
            prec = tp / (tp+fp)    if (tp+fp)    > 0 else float("nan")
            rec  = tp / (tp+fn)    if (tp+fn)    > 0 else float("nan")
            return iou, prec, rec

        iou_b, prec_b, rec_b = _iou_prec_rec(pred_B, mask_peak_B)
        iou_a, prec_a, rec_a = _iou_prec_rec(pred_A, mask_peak_A)

        # Erreur moyenne des offsets en PIXELS (zones supervisées uniquement)
        mask_off = target[:, 4:5]
        n_off    = mask_off.sum()
        if n_off > 0:
            off_err = (pred[:, 2:4] - target[:, 2:4]).abs() * mask_off
            mae_off = (off_err.sum() / (n_off * 2)).item() * OFFSET_NORM
        else:
            mae_off = float("nan")

    return {
        "mse_bg":      mse_bg,
        "mse_stayed":  mse_stayed,
        "iou_b":       iou_b,
        "precision_b": prec_b,
        "recall_b":    rec_b,
        "iou_a":       iou_a,
        "precision_a": prec_a,
        "recall_a":    rec_a,
        "mae_off":     mae_off,
    }


# =============================================================================
# Visualisation
# =============================================================================

def _find_peaks(
    arr: np.ndarray,
    threshold: float = 1.5,
    min_dist: int = 30,
) -> np.ndarray:
    """
    Détecte les maxima locaux au-dessus du seuil.

    Returns : (N, 2) int64 — coordonnées (y, x) des pics détectés.
    """
    if (arr > threshold).sum() == 0:
        return np.zeros((0, 2), dtype=np.int64)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_dist, min_dist))
    dilated = cv2.dilate(arr, kernel)
    peaks   = (arr == dilated) & (arr > threshold)
    return np.argwhere(peaks)


def _colorize_map(
    arr_B: np.ndarray,
    arr_A: np.ndarray,
    threshold: float = 1.5,
) -> np.ndarray:
    """
    Produit une image BGR (H, W, 3) depuis deux cartes float.
        arr_B >= threshold → blanc  (255, 255, 255) — Frame B
        arr_A >= threshold → rouge  (  0,   0, 255) — Frame A
    Frame B est dessiné en dernier (priorité sur Frame A si chevauchement).
    """
    h, w = arr_B.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[arr_A >= threshold] = (0, 0, 255)     # rouge = Frame A
    bgr[arr_B >= threshold] = (255, 255, 255)  # blanc = Frame B (priorité)
    return bgr


def visualize_predictions(
    model_path: str,
    data_root: str,
    output_dir: str,
    img_size: tuple = (1024, 1024),
    n_samples: int = 8,
    base_ch: int = 32,
    split: str = "test",
    indices: list[int] | None = None,
) -> None:
    """
    Charge le modèle et sauvegarde des images de débogage (6 panneaux) :
        Frame A | Frame B | Target (seuil 1.5) | Pred (seuil 1.5) + assoc
        | Pred_B heatmap | Pred_A heatmap
    Sur le panneau Pred, une flèche verte relie chaque pic B détecté à son
    origine prédite dans la Frame A (pic + offset lu aux canaux 2-3).
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

        frame_a   = batch["frame_A"][0].permute(1, 2, 0).numpy()
        frame_b   = batch["frame_B"][0].permute(1, 2, 0).numpy()
        tgt_B_map = target[0, 0].cpu().numpy()
        tgt_A_map = target[0, 1].cpu().numpy()
        pred_B_map = pred[0, 0].cpu().numpy()
        pred_A_map = pred[0, 1].cpu().numpy()
        off_x_map  = pred[0, 2].cpu().numpy() * OFFSET_NORM   # offsets en px
        off_y_map  = pred[0, 3].cpu().numpy() * OFFSET_NORM

        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        def to_bgr(arr):
            arr = np.clip(arr * _std + _mean, 0.0, 1.0)
            return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        def to_heat(arr):
            mn, mx = arr.min(), arr.max()
            normed = (arr - mn) / (mx - mn + 1e-6)
            heat = cv2.applyColorMap((normed * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            return heat, mn, mx

        fa_bgr = to_bgr(frame_a)
        fb_bgr = to_bgr(frame_b)

        tgt_color   = _colorize_map(tgt_B_map,  tgt_A_map,  threshold=1.5)
        pred_thresh = _colorize_map(pred_B_map, pred_A_map, threshold=1.5)

        # Flèches d'association : de chaque pic B vers son origine prédite dans A
        peaks_B = _find_peaks(pred_B_map, threshold=1.5, min_dist=30)
        for (py, px) in peaks_B:
            ox = float(off_x_map[py, px])
            oy = float(off_y_map[py, px])
            pt_a = (int(round(px + ox)), int(round(py + oy)))
            cv2.arrowedLine(pred_thresh, (int(px), int(py)), pt_a,
                            (0, 255, 0), 2, tipLength=0.3)
        pred_heat_B, p_min_B, p_max_B = to_heat(pred_B_map)
        pred_heat_A, p_min_A, p_max_A = to_heat(pred_A_map)

        def add_label(img, text):
            out = img.copy()
            cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return out

        panels = [
            add_label(fa_bgr,       "Frame A"),
            add_label(fb_bgr,       "Frame B"),
            add_label(tgt_color,    "Target (B=blanc A=rouge, seuil 1.5)"),
            add_label(pred_thresh,  "Pred (seuil 1.5) + assoc verte"),
            add_label(pred_heat_B,  f"Pred_B brute [{p_min_B:.2f},{p_max_B:.2f}]"),
            add_label(pred_heat_A,  f"Pred_A brute [{p_min_A:.2f},{p_max_A:.2f}]"),
        ]
        combined = np.concatenate(panels, axis=1)

        # Comptages GT
        gt_A      = sample.get("count_A",      "?")
        gt_B      = sample.get("count_B",      "?")
        gt_stayed = sample.get("count_stayed", "?")

        # Comptage prédit — pics B déjà détectés pour les flèches
        pr_b = len(peaks_B)
        pr_a = len(_find_peaks(pred_A_map, threshold=1.5, min_dist=30))

        header_h = 56
        header   = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
        gt_line   = f"GT :   Frame A={gt_A} pers.  Frame B={gt_B} pers.  [ stayed={gt_stayed} ]"
        pred_line = f"Pred : tetes_B={pr_b} (blanc ch0)   origines_A={pr_a} (rouge ch1)"
        cv2.putText(header, gt_line,   (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(header, pred_line, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255),   1)
        combined = np.vstack([header, combined])

        m = compute_metrics(pred, target)
        stats = (f"iou_B={m['iou_b']:.3f}  prec_B={m['precision_b']:.3f}  rec_B={m['recall_b']:.3f}  "
                 f"iou_A={m['iou_a']:.3f}  prec_A={m['precision_a']:.3f}  rec_A={m['recall_a']:.3f}  "
                 f"mse_bg={m['mse_bg']:.4f}  mae_off={m['mae_off']:.1f}px")
        cv2.putText(combined, stats, (8, combined.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

        out_path = os.path.join(output_dir, f"sample_{ds_idx:04d}.png")
        cv2.imwrite(out_path, combined)
        print(f"  [{sample_num+1}/{len(selected)}] idx={ds_idx} → {out_path}  "
              f"iou_B={m['iou_b']:.3f}  rec_B={m['recall_b']:.3f}  "
              f"iou_A={m['iou_a']:.3f}  rec_A={m['recall_a']:.3f}")

    print(f"\nVisualisations sauvegardées dans : {output_dir}")


# =============================================================================
# Courbes d'entraînement  (layout 2×3)
# =============================================================================

def _plot_training_history(history: dict, save_path: str, best_epoch: int) -> None:
    """Génère et sauvegarde un graphe 2×3 des courbes d'entraînement."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Courbes d'entraînement — V49 (MSE pondérée + offsets)", fontsize=14, fontweight="bold")

    def _vline(ax):
        if best_epoch > 0:
            ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1,
                       label=f"meilleur (ép. {best_epoch})")

    # [0,0] Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train loss", color="steelblue")
    ax.plot(epochs, history["val_loss"],   label="val loss",   color="darkorange")
    _vline(ax); ax.set_title("Loss"); ax.set_xlabel("Époque")
    ax.legend(); ax.grid(True, alpha=0.3)

    # [0,1] Frame B — IoU / Précision / Recall
    ax = axes[0, 1]
    ax.plot(epochs, history["iou_b"],       label="IoU B",       color="purple")
    ax.plot(epochs, history["precision_b"], label="Précision B",  color="steelblue")
    ax.plot(epochs, history["recall_b"],    label="Recall B",     color="green")
    _vline(ax); ax.set_title("Frame B — IoU / Précision / Recall")
    ax.set_xlabel("Époque"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    # [0,2] Frame A — IoU / Précision / Recall  (NOUVEAU)
    ax = axes[0, 2]
    ax.plot(epochs, history["iou_a"],       label="IoU A",       color="coral")
    ax.plot(epochs, history["precision_a"], label="Précision A",  color="orangered")
    ax.plot(epochs, history["recall_a"],    label="Recall A",     color="darkorange")
    _vline(ax); ax.set_title("Frame A — IoU / Précision / Recall")
    ax.set_xlabel("Époque"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    # [1,0] MSE par région
    ax = axes[1, 0]
    ax.plot(epochs, history["mse_bg"],     label="MSE fond",    color="gray")
    ax.plot(epochs, history["mse_stayed"], label="MSE stayed",  color="crimson")
    _vline(ax); ax.set_title("MSE par région")
    ax.set_xlabel("Époque"); ax.set_yscale("log")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")

    # [1,1] Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], color="teal")
    _vline(ax); ax.set_title("Learning rate")
    ax.set_xlabel("Époque"); ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    # [1,2] Erreur moyenne des offsets (px)
    ax = axes[1, 2]
    ax.plot(epochs, history["mae_off"], color="mediumvioletred")
    _vline(ax); ax.set_title("Offsets — erreur moyenne (px)")
    ax.set_xlabel("Époque")
    ax.grid(True, alpha=0.3)

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
    img_size: tuple = (1024, 1024),
    batch_size: int = 4,
    epochs: int = 100,
    lr: float = 1e-3,
    base_ch: int = 32,
    num_workers: int = 0,
    save_path: str = "crowd_tracking_net.pth",
) -> None:
    """Entraîne CrowdTrackingNet V49 (MSE pondérée + offsets CenterTrack)."""
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

    optimizer     = Adam(model.parameters(), lr=lr)
    warmup_epochs = 5
    warmup_sched  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                             total_iters=warmup_epochs)
    cosine_sched  = CosineAnnealingLR(optimizer,
                                      T_max=max(1, epochs - warmup_epochs),
                                      eta_min=1e-5)
    scheduler     = SequentialLR(optimizer,
                                 schedulers=[warmup_sched, cosine_sched],
                                 milestones=[warmup_epochs])

    best_val_loss = float("inf")
    best_epoch    = 0

    history = {
        "train_loss": [], "val_loss": [],
        "iou_b": [], "precision_b": [], "recall_b": [],
        "iou_a": [], "precision_a": [], "recall_a": [],
        "mse_bg": [], "mse_stayed": [], "mae_off": [], "lr": [],
    }

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoque {epoch}/{epochs}  —  LR : {current_lr:.2e}")

        # ---- Entraînement ----
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, 1):
            print(f"  Epoch {epoch} — Batch {batch_idx}/{len(train_loader)}", end="\r")
            x      = torch.cat([batch["frame_A"].to(device), batch["frame_B"].to(device)], dim=1)
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
        agg = {
            "mse_bg": 0.0, "mse_stayed": 0.0,
            "iou_b": 0.0, "precision_b": 0.0, "recall_b": 0.0,
            "iou_a": 0.0, "precision_a": 0.0, "recall_a": 0.0,
            "mae_off": 0.0,
        }
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x      = torch.cat([batch["frame_A"].to(device), batch["frame_B"].to(device)], dim=1)
                target = build_target_batch(batch, img_h, img_w, device)
                pred   = model(x)
                val_loss += tracking_loss(pred, target).item()
                m = compute_metrics(pred, target)
                for k in agg:
                    v = m[k]
                    if v == v:  # ignore NaN
                        agg[k] += v
                        if k == "mse_bg":
                            n_val += 1
        val_loss /= len(val_loader)
        if n_val > 0:
            for k in agg:
                agg[k] /= n_val

        iou_b_v  = agg["iou_b"]       if agg["iou_b"]       == agg["iou_b"]       else 0.0
        prec_b_v = agg["precision_b"] if agg["precision_b"] == agg["precision_b"] else 0.0
        rec_b_v  = agg["recall_b"]    if agg["recall_b"]    == agg["recall_b"]    else 0.0
        iou_a_v  = agg["iou_a"]       if agg["iou_a"]       == agg["iou_a"]       else 0.0
        prec_a_v = agg["precision_a"] if agg["precision_a"] == agg["precision_a"] else 0.0
        rec_a_v  = agg["recall_a"]    if agg["recall_a"]    == agg["recall_a"]    else 0.0

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoque {epoch:>3}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"mse_bg={agg['mse_bg']:.4f}  "
            f"iou_B={iou_b_v:.3f}  prec_B={prec_b_v:.3f}  rec_B={rec_b_v:.3f}  "
            f"iou_A={iou_a_v:.3f}  prec_A={prec_a_v:.3f}  rec_A={rec_a_v:.3f}  "
            f"mae_off={agg['mae_off']:.1f}px  lr={current_lr:.2e}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["iou_b"].append(iou_b_v);        history["precision_b"].append(prec_b_v)
        history["recall_b"].append(rec_b_v)
        history["iou_a"].append(iou_a_v);        history["precision_a"].append(prec_a_v)
        history["recall_a"].append(rec_a_v)
        history["mse_bg"].append(agg["mse_bg"]); history["mse_stayed"].append(agg["mse_stayed"])
        history["mae_off"].append(agg["mae_off"])
        history["lr"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  -> Sauvegarde : {save_path}  (val_loss={val_loss:.4f}  "
                  f"iou_B={iou_b_v:.3f}  rec_B={rec_b_v:.3f}  "
                  f"iou_A={iou_a_v:.3f}  rec_A={rec_a_v:.3f})")

    print(f"\nEntraînement terminé. Meilleur : ép.{best_epoch}  val_loss={best_val_loss:.4f}")
    plot_path = save_path.replace(".pth", "_courbes.png")
    _plot_training_history(history, plot_path, best_epoch)


# =============================================================================
# Smoke test
# =============================================================================

def smoke_test(data_root: str, img_size: tuple = (1024, 1024)) -> None:
    """Vérifie que le modèle V51 tourne correctement sur un batch réel."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_h, img_w = img_size
    print("\n--- Smoke test CrowdTrackingNet (V49) ---")
    print(f"  Appareil : {device}  |  img_size : {img_size}")

    loader, _ = get_tracking_dataloader(
        data_root, split="train", batch_size=2,
        img_size=img_size, augment=False, num_workers=0,
    )
    batch = next(iter(loader))
    x      = torch.cat([batch["frame_A"], batch["frame_B"]], dim=1).to(device)
    target = build_target_batch(batch, img_h, img_w, device)

    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Target : {tuple(target.shape)}  plage=[{target.min():.2f}, {target.max():.2f}]")

    model = CrowdTrackingNet(base_ch=32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x)

    print(f"  Output : {tuple(pred.shape)}")
    assert pred.shape[1] == 4, f"Attendu 4 canaux de sortie, obtenu {pred.shape[1]}"
    assert target.shape[1] == 5, f"Attendu 5 canaux cible, obtenu {target.shape[1]}"
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
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "crowd_tracking_net51.pth")
    VIZ_DIR   = os.path.join(os.path.dirname(__file__), "..", "visualizations/51")

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
            img_size    = (1024, 1024),
            batch_size  = 4,
            epochs      = 100,
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
                img_size   = (1024, 1024),
                n_samples  = 8,
                base_ch    = 32,
                split      = "test",
            )
