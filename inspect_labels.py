import fsspec
import pyarrow.parquet as pq

url = "https://huggingface.co/datasets/timm/nih-chest-xray-14/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

unique_labels = set()
total_rows = 0

print("Scanning labels only (image bytes will not be read)...")

with fsspec.open(url, "rb", block_size=1024 * 1024) as f:
    pf = pq.ParquetFile(f)

    for batch in pf.iter_batches(
        batch_size=500,
        columns=["label_names"]
    ):
        for labels in batch.column("label_names").to_pylist():
            unique_labels.update(labels or [])
            total_rows += 1

print("\nRows scanned:", total_rows)
print("\nUnique disease labels:")
for label in sorted(unique_labels):
    print("-", label)

print("\nTotal unique labels:", len(unique_labels))
