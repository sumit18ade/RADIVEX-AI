from huggingface_hub import HfApi

api = HfApi()
info = api.get_paths_info(
    repo_id="timm/nih-chest-xray-14",
    paths=["default/train/0000.parquet"],
    repo_type="dataset",
    revision="refs/convert/parquet"
)[0]

print("File:", info.path)
print("Size:", round(info.size / (1024**2), 2), "MB")
print("Dataset metadata accessible:", bool(info.size))
print("READY FOR SAMPLE EXTRACTION")
