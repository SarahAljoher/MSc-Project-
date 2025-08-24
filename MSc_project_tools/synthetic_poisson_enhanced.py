
import os
import time
import cv2
import pydicom
import numpy as np
import tifffile
from glob import glob
from scipy.signal.windows import tukey
from getUID import getUID_path
from get_gt  import get_gt

# ← NEW: imports for SAM
import torch
from segment_anything import sam_model_registry, SamPredictor

# ─── CONFIG ────────────────────────────────────────────────────────────────
MSC_BASE     = "/Users/sarahaljoher/Documents/MScdata"
ANNOT_FOLDER = os.path.join(MSC_BASE, "Annotation")
DICOM_BASE   = os.path.join(MSC_BASE, "synthatic_CT_4_train_validation")
HEALTHY_DIR  = os.path.join(MSC_BASE, "healthy_CT")
OUTPUT_DIR   = "./synthetic_poisson_softmask_newtest_"
MASK_DIR     = "./synthetic_poisson_softmask_masks_newtest_newtest"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR,   exist_ok=True)

# ─── UTILITY FUNCTIONS ─────────────────────────────────────────────────────
def normalize_uint8(arr):
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return (((arr - mn) / (mx - mn)) * 255).astype(np.uint8)

def tukey2d(h, w, alpha_h=0.3, alpha_w=0.3):
    wy = tukey(h, alpha=alpha_h)
    wx = tukey(w, alpha=alpha_w)
    return (np.outer(wy, wx) * 255).astype(np.uint8)

# ─── INITIALIZE SAM ─────────────────────────────────────────────────────────
sam_checkpoint = "/Users/sarahaljoher/Documents/MScdata/sam_vit_b_01ec64.pth"
sam_model      = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
device         = "cuda" if torch.cuda.is_available() else "cpu"
sam_model.to(device)
predictor = SamPredictor(sam_model)

# ─── LOAD HEALTHY SCANS ────────────────────────────────────────────────────
healthy_imgs = {}
for path in sorted(glob(os.path.join(HEALTHY_DIR, "*.tif"))):
    raw16   = tifffile.imread(path).astype(np.float32)
    gray8   = normalize_uint8(raw16)
    resized = cv2.resize(gray8, (512, 512), interpolation=cv2.INTER_LINEAR)
    healthy_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    healthy_imgs[path] = healthy_bgr

print("Loaded healthy scans:", [os.path.basename(p) for p in healthy_imgs])

# ─── SUBJECT LOOP ─────────────────────────────────────────────────────────
subjects   = [d for d in os.listdir(DICOM_BASE) if d.startswith("Lung_Dx-")]
total_subj = len(subjects)
print(f" Found {total_subj} subjects to process")
start_time = time.time()

for idx, subj in enumerate(subjects, start=1):
    sid        = subj.replace("Lung_Dx-", "")
    xml_folder = os.path.join(ANNOT_FOLDER, sid)
    dcm_folder = os.path.join(DICOM_BASE, subj)
    if not os.path.isdir(xml_folder):
        continue

    uid_map = getUID_path(dcm_folder)

    for xml in glob(os.path.join(xml_folder, "*.xml")):
        _, boxes = get_gt(xml, num_class=4)
        if boxes.size == 0:
            continue

        uid = os.path.splitext(os.path.basename(xml))[0]
        if uid not in uid_map:
            continue

        dcm_path, _ = uid_map[uid]
        ds          = pydicom.dcmread(dcm_path)
        hu          = ds.pixel_array.astype(np.float32)
        if hu.ndim == 3:
            hu = hu[0]
        if hu.ndim != 2:
            raise RuntimeError(f"Unexpected pixel_array.ndim={hu.ndim} in {dcm_path}")

        for (x1, y1, x2, y2, *_) in boxes.astype(int):
            patch_hu = hu[y1:y2, x1:x2]
            if patch_hu.size == 0:
                print(f" Empty crop for {sid} {uid} bbox {(x1,y1,x2,y2)}, skipping")
                continue

            patch8   = normalize_uint8(patch_hu)
            H, W     = patch8.shape

            softmask = tukey2d(H, W, alpha_h=0.3, alpha_w=0.3)
            src      = cv2.cvtColor(patch8, cv2.COLOR_GRAY2BGR)
            center   = (x1 + W // 2, y1 + H // 2)

            for hp, dst_full in healthy_imgs.items():
                hH, hW = dst_full.shape[:2]
                if x2 > hW or y2 > hH:
                    continue

                # ─── blending identical to before ───────────────────────────────
                blended  = cv2.seamlessClone(
                    src=src,
                    dst=dst_full,
                    mask=softmask,
                    p=center,
                    flags=cv2.NORMAL_CLONE
                )
                gray_out = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)

                out_name = f"{sid}_{uid}_{os.path.splitext(os.path.basename(hp))[0]}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), gray_out)

                # ─── NEW MASK via SAM + closing ────────────────────────────────
                rgb_out = cv2.cvtColor(gray_out, cv2.COLOR_GRAY2RGB)
                predictor.set_image(rgb_out)
                masks, scores, logits = predictor.predict(
                    box=np.array([[x1, y1, x2, y2]]),
                    multimask_output=False
                )
                # take the single mask, convert to uint8
                mask_bin = (masks[0].astype(np.uint8)) * 255

                # closing to smooth the mask
                kernel = np.ones((20, 20), np.uint8)
                mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

                cv2.imwrite(os.path.join(MASK_DIR, out_name), mask_bin)

    elapsed = time.time() - start_time
    avg     = elapsed / idx
    eta     = (total_subj - idx) * avg
    print(f"[{idx}/{total_subj}] ✔ Done subject {sid} – elapsed {elapsed:.1f}s, ETA {eta:.1f}s")

print(" Finished synthetic blending.")
print("   Images saved in:", OUTPUT_DIR)
print("   SAM-based masks in:", MASK_DIR)
