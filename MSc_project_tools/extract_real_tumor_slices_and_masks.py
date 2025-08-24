import os
import time
import cv2
import numpy as np
import pydicom
from glob import glob
import torch
from segment_anything import sam_model_registry, SamPredictor
from getUID import getUID_path
from get_gt  import get_gt

# ─── 0) CONFIG ─────────────────────────────────────────────────────────────

ANNOT_DIR   = "/Users/sarahaljoher/Documents/MScdata/Annotation"
OUT_BASE    = "/Users/sarahaljoher/Documents/MScdata/VisualizationTools"

# #Uncomment to generate real slices for training and validation 
# REAL_DIR    = "/Users/sarahaljoher/Documents/MScdata/real_CT_4_train_validation"
# IMG_OUT     = os.path.join(OUT_BASE, "real_slices_png")
# MSK_OUT     = os.path.join(OUT_BASE, "real_slices_masks")

#Uncomment to generate real slices for test 
REAL_DIR    = "/Users/sarahaljoher/Documents/MScdata/real_CT_4_test2"
IMG_OUT     = os.path.join(OUT_BASE, "real_test_slices_png2")
MSK_OUT     = os.path.join(OUT_BASE, "real_test_slices_masks2")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(MSK_OUT, exist_ok=True)

# ─── 1) INIT SAM ────────────────────────────────────────────────────────────
SAM_CHECKPOINT = "/Users/sarahaljoher/Documents/MScdata/sam_vit_b_01ec64.pth"
device   = "cuda" if torch.cuda.is_available() else "cpu"
sam      = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(device)
predictor= SamPredictor(sam)

# ─── 2) HELPERS ─────────────────────────────────────────────────────────────
def normalize_uint8(arr):
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return (((arr - mn)/(mx - mn)) * 255.0).astype(np.uint8)

# ─── 3) COLLECT SUBJECTS & START TIMER ──────────────────────────────────────
subjects    = [d for d in os.listdir(REAL_DIR) if d.startswith("Lung_Dx-")]
total_subj  = len(subjects)
print(f" Found {total_subj} subjects to process")
start_time  = time.time()

# ─── 4) MAIN LOOP ───────────────────────────────────────────────────────────
for idx, subj in enumerate(subjects, start=1):
    subj_id    = subj.replace("Lung_Dx-","")
    dcm_folder = os.path.join(REAL_DIR, subj)
    xml_folder = os.path.join(ANNOT_DIR,  subj_id)
    if not os.path.isdir(xml_folder):
        # skip missing annotations
        continue

    # map DICOM UIDs → file paths
    uid_map = getUID_path(dcm_folder)

    # process each XML under this subject
    for xml_path in glob(os.path.join(xml_folder, "*.xml")):
        # read bounding boxes
        _, boxes = get_gt(xml_path, num_class=4)
        if boxes.size == 0:
            continue

        uid = os.path.splitext(os.path.basename(xml_path))[0]
        if uid not in uid_map:
            continue

        # load DICOM slice once
        dcm_path, _ = uid_map[uid]
        ds          = pydicom.dcmread(dcm_path)
        hu          = ds.pixel_array.astype(np.float32)
        img8        = normalize_uint8(hu)            # H×W uint8
        H, W        = img8.shape

        # save slice PNG
        slice_name  = f"{subj_id}_{uid}.png"
        cv2.imwrite(os.path.join(IMG_OUT, slice_name), img8)

        # run SAM on that RGB image
        rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        predictor.set_image(rgb)

        # accumulate per-box masks
        full_mask = np.zeros((H,W), dtype=np.uint8)
        for (x1,y1,x2,y2,*_) in boxes.astype(int):
            box_np = np.array([[x1,y1,x2,y2]])
            masks, _, _ = predictor.predict(
                box=box_np,
                multimask_output=False
            )
            m = (masks[0].astype(np.uint8))*255
            full_mask = np.maximum(full_mask, m)

        # closing to fill small holes
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

        # save mask PNG
        cv2.imwrite(os.path.join(MSK_OUT, slice_name), full_mask)

    # ─── progress & ETA ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    avg     = elapsed / idx
    rem     = total_subj - idx
    eta     = rem * avg
    print(f"[{idx}/{total_subj}] ✔ Done subject {subj_id} "
          f"– elapsed {elapsed:.1f}s, ETA {eta:.1f}s")

print(" All done.")
print(" • Slices →", IMG_OUT)
print(" • Masks  →", MSK_OUT)
