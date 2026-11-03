from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

print("Downloading one dataset shard (~501 MB)...")

path = hf_hub_download(
    repo_id="timm/nih-chest-x-ray-14",
    filename="default/train/0000.parquet",
    repo_type="dataset",
    revision="refs/convert/parquet"
)

print("\nDownload completed!")
parquet_file = pq.ParquetFile(path)

print("\nTotal rows:", parquet_file.metadata.num_rows)
print("Row groups:", parquet_file.metadata.num_row_groups)
print("\nDataset columns and schema:")
print(parquet_file.schema_arrow)
