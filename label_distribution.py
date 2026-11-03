import fsspec
import pyarrow.parquet as pq
from collections import Counter

url = "https://huggingface.co/datasets/timm/nih-chest-xray-14/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

counts = Counter()
total = 0
no_finding = 0

with fsspec.open(url, "rb", block_size=1024 * 1024) as f:
    pf = pq.ParquetFile(f)

    for batch in pf.iter_batches(batch_size=500, columns=["label_names"]):
        for labels in batch.column("label_names").to_pylist():
            total += 1
            if not labels:
                no_finding += 1
            else:
                counts.update(labels)

print("Total records:", total)
print("No Finding records (empty labels):", no_finding)
print("\nDisease-wise image counts:")

for label, count in sorted(counts.items()):
    print(f"{label}: {count}")

print("\nTotal disease labels:", len(counts))
