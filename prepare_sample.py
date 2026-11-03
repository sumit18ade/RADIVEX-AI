import cv2
import pandas as pd
from pathlib import Path

source = Path("sample_data/images")
target = Path("sample_data/CXR14/preprocessed224")
target.mkdir(parents=True, exist_ok=True)

classes = [
    "cardiomegaly", "effusion", "pneumonia", "atelectasis",
    "no_finding", "consolidation", "pneumothorax", "fibrosis",
    "emphysema", "hernia", "pleural_thickening", "edema",
    "nodule", "mass"
]

rows = []

for index, path in enumerate(sorted(source.glob("*.png")), 1):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Skipped:", path.name)
        continue

    image = cv2.resize(image, (224, 224))
    cv2.imwrite(str(target / path.name), image)

    row = {
        "image_id": path.stem,
        "img_rel_path": path.name,
        "split": "train"
    }

    # Labels are populated in the next step from source metadata.
    for label in classes:
        row[label] = 0

    rows.append(row)
    print(f"Processed {index}: {path.name}")

pd.DataFrame(rows).to_csv("sample_data/train.csv", index=False)

print("\nPREPROCESSING COMPLETE")
print("Images:", len(rows))
print("CSV: sample_data/train.csv")
print("Image folder:", target)
