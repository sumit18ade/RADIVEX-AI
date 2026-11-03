from data.data_provider import cxr14_data_loader_2D
from torch.utils.data import DataLoader

CONFIG_PATH = "config/config.yaml"

print("Loading dataset...")

train_dataset = cxr14_data_loader_2D(
    cfg_path=CONFIG_PATH,
    mode="train",
    augment=False,
    image_size=224
)

print("Dataset loaded successfully!")
print("Training samples:", len(train_dataset))

if len(train_dataset) > 0:
    image, label = train_dataset[0]

    print("\nSample verification:")
    print("Image shape:", image.shape)
    print("Label shape:", label.shape)
    print("Label values:", label.tolist())

    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    images, labels = next(iter(loader))

    print("\nBatch verification:")
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)

    print("\nDATALOADER TEST PASSED!")
else:
    print("ERROR: Training dataset is empty.")