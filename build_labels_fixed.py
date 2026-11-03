import fsspec
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

url = "https://huggingface.co/datasets/timm/nih-chest-xray-14/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

image_dir = Path("sample_data/CXR14/preprocessed224")
image_ids = {p.stem for p in image_dir.glob("*.png")}

classes = [
    "cardiomegaly", "effusion", "pneumonia", "atelectasis",
    "no_finding", "consolidation", "pneumothorax", "fibrosis",
    "emphysema", "hernia", "pleural_thickening", "edema",
    "nodule", "mass"
]

rows = []

with fsspec.open(url, "rb", block_size=1024 * 1024) as f:
    pf = pq.ParquetFile(f)

    for batch in pf.iter_batches(
        batch_size=100,
        columns=["image_id", "label_names"]
    ):
        for item in batch.to_pylist():
            image_id = item["image_id"]

            if image_id not in image_ids:
                continue

            labels = item["label_names"] or []

            # Infiltration is not an output class in this project.
            if "Infiltration" in labels:
                continue

            normalized = {x.lower() for x in labels}

            row = {
                "image_id": image_id,
                "img_rel_path": image_id + ".png",
                "split": "train"
            }

            for cls in classes:
                row[cls] = int(
                    not labels if cls == "no_finding"
                    else cls in normalized
                )

            rows.append(row)

df = pd.DataFrame(rows)
output = Path("sample_data/final_cxr14_master_list.csv")
df.to_csv(output, index=False)

print("FINAL CSV CREATED")
print("Usable images:", len(df))
print("Excluded Infiltration images:", len(image_ids) - len(df))
print("\nLabel totals:")
print(df[classes].sum().to_string())
print("\nSaved:", output)
