import pandas as pd
from pathlib import Path

csv_path = Path("sample_data/final_cxr14_master_list.csv")
image_dir = Path("sample_data/CXR14/preprocessed224")

df = pd.read_csv(csv_path)

classes = [
    "cardiomegaly", "effusion", "pneumonia", "atelectasis",
    "no_finding", "consolidation", "pneumothorax", "fibrosis",
    "emphysema", "hernia", "pleural_thickening", "edema",
    "nodule", "mass"
]

missing = [
    name for name in df["img_rel_path"]
    if not (image_dir / name).exists()
]

print("CSV rows:", len(df))
print("Missing images:", len(missing))
print("Duplicate IDs:", df["image_id"].duplicated().sum())
print("Label columns:", len(classes))
print("Image dimensions:", __import__("cv2").imread(
    str(image_dir / df.iloc[0]["img_rel_path"])
).shape)

print("\nVALIDATION:", "PASSED" if not missing and not df["image_id"].duplicated().any() else "FAILED")
