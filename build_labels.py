import fsspec
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

url = "https://huggingface.co/datasets/timm/nih-chest-xray-14/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

classes = [
    "cardiomegaly", "effusion", "pneumonia", "atelectasis",
    "no_finding", "consolidation", "pneumothorax", "fibrosis",
    "emphysema", "hernia", "pleural_thickening", "edema",
    "nodule", "mass"
]

image_dir = Path("sample_data/CXR14/preprocessed224")
rows = []

print("Reading original labels from remote dataset...")

with fsspec.open(url, "rb", block_size=1024 * 1024) as f:
    pf = pq.ParquetFile(f)

    for batch in pf.iter_batches(
        batch_size=100,
        columns=["image_id", "label_names"]
    ):
        for item in batch.to_pylist():
            image_id = item["image_id"]
            labels = item["label_names"] or []

            if not (image_dir / f"{image_id}.png").exists():
                continue

            if "Infiltration" in labels:
                continue

            row = {
                "image_id": image_id,
                "img_rel_path": f"{image_id}.png",
                "split": "train"
            }

            for cls in classes:
                if cls == "no_finding":
                    row[cls] = int(len(labels) == 0)
                else:
                    row[cls] = int(cls in labels)

            rows.append(row)

            if len(rows) >= 100:
                break

        if len(rows) >= 100:
            break

output = Path("sample_data/final_cxr14_master_list.csv")
pd.DataFrame(rows).to_csv(output, index=False)

print("\nLABEL MAPPING COMPLETE")
print("Usable images:", len(rows))
print("Excluded Infiltration records from this sample.")
print("CSV:", output)
print("Columns:", list(pd.DataFrame(rows).columns))
