# import os

# # Use the subject-level folder containing DICOM subfolders
# dicom_path = "/Users/sarahaljoher/Documents/MScdata/Lung_Dx-A0032"

# # Path to folder containing the XML annotation(s) for that subject
# annotation_path = "/Users/sarahaljoher/Documents/MScdata/Annotation/A0032"

# # Class labels file
# classfile = "/Users/sarahaljoher/Documents/MScdata/VisualizationTools/category.txt"

# # Run the visualizer script using os.system
# os.system(
#     f'python visualization.py --dicom-mode CT --dicom-path "{dicom_path}" --annotation-path "{annotation_path}" --classfile "{classfile}"'
# )



# ###########
# import os
# from getUID import getUID_path
# from get_gt import get_gt
# from extract_tumor_from_bbox import extract_tumor_from_bbox

# #  Set paths
# dicom_folder = "/Users/sarahaljoher/Documents/MScdata/Lung_Dx-A0010"
# xml_folder = "/Users/sarahaljoher/Documents/MScdata/Annotation/A0010"
# subject_name = os.path.basename(xml_folder)  # e.g. "A0032"

# #  Step 1: Map all UIDs from DICOM folder
# uid_dict = getUID_path(dicom_folder)

# #  Step 2: Loop through all XML files in this annotation folder
# xml_files = [f for f in os.listdir(xml_folder) if f.endswith(".xml")]

# for xml_file in xml_files:
#     uid = xml_file.replace(".xml", "")
    
#     if uid not in uid_dict:
#         print(f"Skipping {uid}: not found in DICOM folder")
#         continue

#     xml_path = os.path.join(xml_folder, xml_file)
#     dcm_path, dcm_name = uid_dict[uid]
#     full_dicom_path = dcm_path  # Already includes filename

#     # Step 3: Get bounding box from XML
#     _, img_data = get_gt(xml_path, num_class=4)

#     if len(img_data) == 0:
#         print(f" No bounding box in {xml_file}")
#         continue

#     bbox = list(map(int, img_data[0][:4]))

#     #  Step 4: Extract and save tumor
#     print(f"✔️ Extracting tumor from DICOM: {full_dicom_path}")
#     print(f"   → Bounding box: {bbox}")
#     # extract_tumor_from_bbox(full_dicom_path, bbox, output_dir="./tumor_output")
#     extract_tumor_from_bbox(
#     dicom_path=full_dicom_path,
#     bbox=bbox,
#     output_dir="./tumor_output",
#     subject=subject_name
#     )


# import os
# from getUID import getUID_path
# from get_gt import get_gt
# from extract_tumor_from_bbox import extract_tumor_from_bbox

# # Base directories
# mscdata_base = "/Users/sarahaljoher/Documents/MScdata/real_CT_4_train_validation"
# annotation_base = os.path.join(mscdata_base, "Annotation")
# output_base = os.path.join(mscdata_base, "tumor_output")

# # List all DICOM folders starting with Lung_Dx-
# dicom_folders = [f for f in os.listdir(mscdata_base) if f.startswith("Lung_Dx-")]
# print(f" Found {len(dicom_folders)} subjects.")

# for dicom_folder_name in dicom_folders:
#     subject_id = dicom_folder_name.replace("Lung_Dx-", "")  # e.g., "A0032"
#     dicom_folder = os.path.join(mscdata_base, dicom_folder_name)
#     xml_folder = os.path.join(annotation_base, subject_id)

#     if not os.path.isdir(xml_folder):
#         print(f" No annotation folder for subject {subject_id}, skipping.")
#         continue

#     # Build UID-to-DICOM mapping
#     try:
#         uid_dict = getUID_path(dicom_folder)
#     except Exception as e:
#         print(f"⚠ Failed to load DICOM for {subject_id}: {e}")
#         continue

#     # Process each XML in this subject
#     xml_files = [f for f in os.listdir(xml_folder) if f.endswith(".xml")]
#     if not xml_files:
#         print(f" No XML files in {xml_folder}")
#         continue

#     for xml_file in xml_files:
#         uid = xml_file.replace(".xml", "")
#         if uid not in uid_dict:
#             print(f"Skipping {uid} — not found in DICOM folder")
#             continue

#         xml_path = os.path.join(xml_folder, xml_file)
#         dcm_path, dcm_name = uid_dict[uid]
#         full_dicom_path = dcm_path  # already includes the filename

#         # Load XML annotation
#         _, img_data = get_gt(xml_path, num_class=4)
#         if len(img_data) == 0:
#             print(f"No bounding box in {xml_file}, skipping.")
#             continue

#         bbox = list(map(int, img_data[0][:4]))

#         # Output path for this subject
#         subject_output = os.path.join(output_base, subject_id)

#         # Run extraction
#         print(f" Extracting from {subject_id} → {os.path.basename(full_dicom_path)}")
#         extract_tumor_from_bbox(
#             dicom_path=full_dicom_path,
#             bbox=bbox,
#             output_dir=subject_output,
#             subject=subject_id
#         )

# import os
# from getUID import getUID_path
# from get_gt import get_gt
# from extract_tumor_from_bbox import extract_tumor_from_bbox

# # Base directories
# data_base = "/Users/sarahaljoher/Documents/MScdata/synthatic_CT_4_train_validation/untitled folder 2"
# annotation_base = "/Users/sarahaljoher/Documents/MScdata/Annotation"
# output_base = os.path.join(data_base, "tumor_output")

# # List all DICOM folders starting with Lung_Dx-
# dicom_folders = [f for f in os.listdir(data_base) if f.startswith("Lung_Dx-")]
# print(f" Found {len(dicom_folders)} subjects.")

# for dicom_folder_name in dicom_folders:
#     subject_id = dicom_folder_name.replace("Lung_Dx-", "")  # e.g., "A0032"
#     dicom_folder = os.path.join(data_base, dicom_folder_name)
#     xml_folder = os.path.join(annotation_base, subject_id)

#     if not os.path.isdir(xml_folder):
#         print(f" ❌ No annotation folder for subject {subject_id}, skipping.")
#         continue

#     try:
#         uid_dict = getUID_path(dicom_folder)
#     except Exception as e:
#         print(f"⚠️ Failed to load DICOM for {subject_id}: {e}")
#         continue

#     xml_files = [f for f in os.listdir(xml_folder) if f.endswith(".xml")]
#     if not xml_files:
#         print(f" ⚠️ No XML files in {xml_folder}")
#         continue

#     for xml_file in xml_files:
#         uid = xml_file.replace(".xml", "")
#         if uid not in uid_dict:
#             print(f" Skipping {uid} — not found in DICOM folder")
#             continue

#         xml_path = os.path.join(xml_folder, xml_file)
#         dcm_path, dcm_name = uid_dict[uid]

#         _, img_data = get_gt(xml_path, num_class=4)
#         if len(img_data) == 0:
#             print(f" No bounding box in {xml_file}, skipping.")
#             continue

#         bbox = list(map(int, img_data[0][:4]))

#         # Create subject output folder if not exists
#         subject_output = os.path.join(output_base, subject_id)
#         os.makedirs(subject_output, exist_ok=True)

#         print(f" 🔍 Extracting: {subject_id} – {os.path.basename(dcm_path)}")

#         extract_tumor_from_bbox(
#             dicom_path=dcm_path,
#             bbox=bbox,
#             output_dir=subject_output,
#             subject=f"{subject_id}_{uid}"  # make subject ID unique per slice
#         )




import os
import numpy as np
import pydicom
from PIL import Image, ImageDraw

from getUID import getUID_path
from get_gt import get_gt

# Base directories
data_base       = "/Users/sarahaljoher/Documents/MScdata/real_CT_4_test2"
annotation_base = "/Users/sarahaljoher/Documents/MScdata/Annotation"
output_base     = os.path.join(data_base, "bbox_overlays")


# Gather all DICOM subject folders
dicom_folders = [f for f in os.listdir(data_base) if f.startswith("Lung_Dx-")]
print(f"Found {len(dicom_folders)} subjects.")

for dicom_folder_name in dicom_folders:
    subject_id   = dicom_folder_name.replace("Lung_Dx-", "")
    dicom_folder = os.path.join(data_base, dicom_folder_name)
    xml_folder   = os.path.join(annotation_base, subject_id)

    if not os.path.isdir(xml_folder):
        print(f"❌ No annotation for {subject_id}, skipping.")
        continue

    try:
        uid_dict = getUID_path(dicom_folder)  # returns {uid: (dcm_path, dcm_name), ...}
    except Exception as e:
        print(f"⚠️ Failed loading DICOM for {subject_id}: {e}")
        continue

    xml_files = [f for f in os.listdir(xml_folder) if f.endswith(".xml")]
    if not xml_files:
        print(f"⚠️ No XML in {xml_folder}, skipping.")
        continue

    # Prepare output folder for this subject
    subject_out = os.path.join(output_base, subject_id)
    os.makedirs(subject_out, exist_ok=True)

    for xml_file in xml_files:
        uid = xml_file[:-4]  # strip ".xml"
        if uid not in uid_dict:
            print(f"Skipping slice {uid} (not in DICOM).")
            continue

        xml_path    = os.path.join(xml_folder, xml_file)
        dcm_path, _ = uid_dict[uid]

        # Get ground-truth boxes
        _, img_data = get_gt(xml_path, num_class=4)
        if len(img_data) == 0:
            print(f"No bbox in {xml_file}, skipping.")
            continue

        # Use the first bbox: [x_min, y_min, x_max, y_max]
        x0, y0, x1, y1 = map(int, img_data[0][:4])

        # Read the correct DICOM file and normalize pixel values to 0–255
        ds  = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(float)
        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)

        # Create an RGB image and draw the bbox
        img  = Image.fromarray(arr).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

        # Save the overlay image
        out_fname = f"{subject_id}_{uid}_bbox.png"
        out_path  = os.path.join(subject_out, out_fname)
        img.save(out_path)

        print(f"✅ Saved overlay: {out_path}")
