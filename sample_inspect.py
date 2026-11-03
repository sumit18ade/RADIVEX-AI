import fsspec
import pyarrow.parquet as pq

url = "https://huggingface.co/datasets/timm/nih-chest-xray-14/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

with fsspec.open(url, "rb", block_size=1024 * 1024) as f:
    pf = pq.ParquetFile(f)

    print("FIRST 10 DATASET RECORDS\n")

    for batch in pf.iter_batches(batch_size=10):
        data = batch.to_pylist()

        for i, row in enumerate(data, 1):
            image = row["image"]
            print(f"Record {i}")
            print("Image ID:", row["image_id"])
            print("Labels:", row["label_names"])
            print("Image bytes:", len(image["bytes"]) if image["bytes"] else 0)
            print("Patient ID:", row["patient_id"])
            print("View:", row["view_position"])
            print("-" * 40)
        break
