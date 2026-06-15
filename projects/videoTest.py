"""
videoTest.py — Test des modèles sur une VIDÉO (rendu annoté temps réel)
=======================================================================

Applique un modèle entraîné sur une vidéo et écrit UNE SEULE vidéo annotée
(annotations dessinées image par image → rendu « temps réel »).

Choix du modèle via la variable MODEL (en bas du fichier) :

  "V51"     — CrowdTrackingNet (U-Net heatmap + offsets). Pour CHAQUE paire de
              frames consécutives (i, i+1) = (Frame A, Frame B), le modèle
              détecte les têtes de la frame COURANTE (B = i+1) et, via les
              offsets, retrouve leur position dans la frame PRÉCÉDENTE
              (A = i). On dessine :
                • un point sur chaque tête courante,
                • une flèche depuis sa position précédente → suivi du mouvement.

  "V50YOLO" — détecteur de boîtes YOLO (têtes). Détection image par image sur
              la frame courante (i+1). YOLO ne fait pas d'association entre
              frames : on dessine juste les boîtes détectées.

Pour les deux : Frame A = frame i, Frame B = frame i+1 (paires consécutives).

Sortie : ../annotated_<MODEL>_<nom_video>.mp4 (à la racine du repo).

Dépendances : torch (V51), ultralytics (V50YOLO), opencv-python.
"""

import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))   # projects/
_ROOT = os.path.dirname(_HERE)                        # racine du repo
sys.path.insert(0, _HERE)

# Normalisation ImageNet (identique à l'entraînement de CrowdTrackingNet)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =============================================================================
# Backend V51 — CrowdTrackingNet (heatmap + offsets)
# =============================================================================

class _V51Backend:
    """Encapsule le modèle V51 : pré-traitement, inférence par paire, dessin."""

    def __init__(self, weights: str, img_size: int = 1024,
                 peak_thres: float = 1.5, min_dist: int = 30, base_ch: int = 32):
        import modelcopyV51 as net   # importe l'architecture + constantes
        self._net = net
        self.img_size   = img_size
        self.peak_thres = peak_thres
        self.min_dist   = min_dist
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.isfile(weights):
            raise FileNotFoundError(f"Checkpoint V51 introuvable : {weights}")
        self.model = net.CrowdTrackingNet(base_ch=base_ch).to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval()
        print(f"[V51] Modèle chargé : {weights}  (device={self.device})")

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Frame BGR → tenseur (3, S, S) normalisé ImageNet."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        return torch.from_numpy(rgb.transpose(2, 0, 1))

    def annotate(self, frame_A: np.ndarray, frame_B: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Annote frame_B (courante) à partir de la paire (A=i, B=i+1).
        Retourne (image annotée, nb de têtes détectées).
        """
        H, W = frame_B.shape[:2]
        a = self._preprocess(frame_A)
        b = self._preprocess(frame_B)
        x = torch.cat([a, b], dim=0).unsqueeze(0).to(self.device)  # (1, 6, S, S)

        with torch.no_grad():
            pred = self.model(x)
        hm_B  = pred[0, 0].cpu().numpy()
        off_x = pred[0, 2].cpu().numpy() * self._net.OFFSET_NORM
        off_y = pred[0, 3].cpu().numpy() * self._net.OFFSET_NORM

        peaks = self._net._find_peaks(hm_B, threshold=self.peak_thres, min_dist=self.min_dist)

        sx, sy = W / self.img_size, H / self.img_size
        out = frame_B.copy()
        for (py, px) in peaks:
            # position courante (frame B) en coordonnées d'origine
            cx, cy = int(px * sx), int(py * sy)
            # position précédente (frame A) = pic + offset
            ax = int((px + float(off_x[py, px])) * sx)
            ay = int((py + float(off_y[py, px])) * sy)
            # flèche : position précédente → position courante (mouvement)
            cv2.arrowedLine(out, (ax, ay), (cx, cy), (0, 255, 0), 1, tipLength=0.35)
            cv2.circle(out, (cx, cy), 3, (255, 255, 255), -1)  # tête courante
        return out, len(peaks)


# =============================================================================
# Backend V50YOLO — détecteur de boîtes
# =============================================================================

class _YOLOBackend:
    """Encapsule le détecteur YOLO via les helpers de modelcopyV50YOLO."""

    def __init__(self, target: str, conf: float = 0.25, img_size: int = 1280):
        import modelcopyV50YOLO as yolo
        self._yolo = yolo
        self.conf = conf
        self.img_size = img_size

        weights, target_class, self.label = yolo.resolve_target(target)
        self.model = yolo._load_model(weights)
        if self.model is None:
            raise FileNotFoundError(f"Modèle YOLO indisponible pour la cible : {target}")
        self.cls_id, self.cls_name = yolo._resolve_class(self.model, target_class)
        print(f"[V50YOLO] cible={target}  classe={self.cls_id} ({self.cls_name})")

    def annotate(self, frame_A: np.ndarray, frame_B: np.ndarray) -> tuple[np.ndarray, int]:
        """YOLO détecte sur la frame COURANTE (B). frame_A non utilisée."""
        boxes = self._yolo._detect_boxes(self.model, frame_B, self.cls_id,
                                         self.conf, self.img_size)
        out = frame_B.copy()
        for (x1, y1, x2, y2) in boxes.astype(int):
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return out, len(boxes)


# =============================================================================
# Boucle vidéo
# =============================================================================

def run_on_video(
    backend,
    video_in: str,
    video_out: str,
    label: str,
    max_frames: int = 0,
    frame_stride: int = 1,
) -> None:
    """
    Parcourt la vidéo par paires consécutives (i, i+1), annote chaque frame
    courante et écrit la vidéo de sortie.

    max_frames  : 0 = toute la vidéo, sinon limite (test rapide).
    frame_stride: 1 = toutes les frames ; n = une frame sur n (accélère).
    """
    if not os.path.isfile(video_in):
        print(f"[ERREUR] Vidéo introuvable : {video_in}")
        return

    # Crée le dossier de sortie (ex. videos/trained/) s'il n'existe pas encore,
    # sinon cv2.VideoWriter échoue silencieusement.
    os.makedirs(os.path.dirname(os.path.abspath(video_out)), exist_ok=True)

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo : {video_in}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = max(1.0, fps / frame_stride)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Vidéo : {video_in}")
    print(f"  {total} frames @ {fps:.1f} fps  |  stride={frame_stride}  |  max={max_frames or 'all'}")

    writer = None
    prev = None
    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride != 0:
            idx += 1
            continue

        if prev is not None:
            annotated, n_det = backend.annotate(prev, frame)  # (A=prev, B=courante)

            # Bandeau d'info (annotation « temps réel »)
            bar = annotated.copy()
            txt = f"{label}  frame {idx}  detections={n_det}"
            cv2.rectangle(bar, (0, 0), (bar.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(bar, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            annotated = bar

            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_out, fourcc, out_fps, (w, h))
            writer.write(annotated)
            written += 1
            if written % 25 == 0:
                print(f"  ... {written} frames annotées (idx={idx})", end="\r")

        prev = frame
        idx += 1
        if max_frames and written >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nVidéo annotée écrite : {video_out}  ({written} frames)")
    else:
        print("[ERREUR] Aucune frame écrite (vidéo trop courte ?).")


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    # --- Choix du modèle : "V51" ou "V50YOLO" --------------------------------
    MODEL = "V51"   # "V51" | "V50YOLO"

    # --- Vidéo d'entrée (dans la racine du repo) -----------------------------
    # Plusieurs vidéos de test possibles (téléchargées depuis Pexels, voir README) :
    
    # background video _ people _ walking _.mp4  (leger)
    # People Walking Past the Camera - Free Stock Footage For Commercial Projects.mp4 (lourd)
    # People Walking Free Stock Footage, Royalty-Free No Copyright Content.mp4 (lourd)
    # Crowd Video Stock Footage.mp4 (leger)
    # IMAGES _ Manifestation à Genève à la veille du sommet du G7.mp4 (lourd)
    # Massive strike and protest slam ICE bullying in Minneapolis.mp4 (lourd)
    # Times Square Crowd People 2 HD Video Background.mp4 (leger)
    
    VIDEO_NAME = "videos/source/Times Square Crowd People 2 HD Video Background.mp4"
    VIDEO_IN   = os.path.join(_ROOT, VIDEO_NAME)

    # --- Options communes ----------------------------------------------------
    MAX_FRAMES   = 0    # 0 = toute la vidéo ; sinon limite (ex. 100 pour un test rapide)
    FRAME_STRIDE = 1    # 1 = toutes les frames ; 2 = une sur deux (plus rapide)

    # --- Paramètres spécifiques ----------------------------------------------
    V51_WEIGHTS  = os.path.join(_HERE, "crowd_tracking_net51.pth")
    V51_IMG_SIZE = 1024
    V51_PEAK_THRES = 1.5

    YOLO_TARGET  = "head_finetuned"   # "head_finetuned" | "head_external" | "person_coco"
    YOLO_CONF    = 0.25

    stem = os.path.splitext(os.path.basename(VIDEO_IN))[0][:40].strip().replace(" ", "_")
    VIDEO_OUT = os.path.join(_ROOT, f"videos/trained/annotated_{MODEL}_{stem}.mp4")

    if MODEL == "V51":
        backend = _V51Backend(V51_WEIGHTS, img_size=V51_IMG_SIZE, peak_thres=V51_PEAK_THRES)
        label = "V51 (heatmap+offsets)"
    elif MODEL == "V50YOLO":
        backend = _YOLOBackend(YOLO_TARGET, conf=YOLO_CONF)
        label = f"V50YOLO ({YOLO_TARGET})"
    else:
        raise ValueError(f"MODEL inconnu : {MODEL!r} (attendu 'V51' ou 'V50YOLO')")

    run_on_video(backend, VIDEO_IN, VIDEO_OUT, label,
                 max_frames=MAX_FRAMES, frame_stride=FRAME_STRIDE)
