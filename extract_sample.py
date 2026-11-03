import fsspec
import pyarrow.parquet as pq
from pathlib import Path

url = "https://huggingface.co/datasets/timm/nih-chest-xray-14/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

out = Path("sample_data/images")
out.mkdir(parents=True, exist_ok=True)

saved = 0

with fsspec.open(url, "rb", block_size=1024 * 1024) as f:
    pf = pq.ParquetFile(f)

    for batch in pf.iter_batches(
        batch_size=10,
        columns=["image", "image_id", "label_names", "patient_id", "view_position"]
    ):
        for row in batch.to_pylist():
            if saved >= 100:
                break

            image_bytes = row["image"]["bytes"]
            if not image_bytes:
                continue

            filename = row["image_id"] + ".png"
            (out / filename).write_bytes(image_bytes)
            saved += 1

            print(f"Saved {saved}/100: {filename}")

        if saved >= 100:
            break

print(f"\nDONE! Images saved: {saved}")
print("Folder:", out.resolve())
