import csv
from config import dataset_paths, checkpoint_path
from dataset import HybridDataset
import numpy as np
from model import SegmentationModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd

# negative, intact, wildtype are labeled as 0, 0, 0
output_paths = {
    "train_val": "features.csv",
    "train": "features-train.csv",
    "val": "features-val.csv",
    "test": "features-test.csv",
}
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    for key in ["train", "val", "test"]:
        extract_features_from_dataset(
            HybridDataset(*dataset_paths, split=key), output_paths[key]
        )
    combine_csv_rows(
        output_paths["train"], output_paths["val"], output_paths["train_val"]
    )


def extract_features_from_dataset(dataset, output_csv):
    model = SegmentationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.to(device)
    model.eval()

    all_rows = []

    metadata_keys = list(dataset[0][2].keys())

    for image, mask, metadata in dataset:
        deep_features = extract_deep_features(image, mask, model)  # [1, C*2]
        features_vector = deep_features.squeeze(0).cpu().numpy()  # [C*2]

        # encode metadata numerically
        encoded_metadata = encode_metadata({k: metadata[k] for k in metadata_keys})

        # concatenate features + encoded metadata
        row = np.concatenate(
            [features_vector, np.array(encoded_metadata, dtype=np.float32)]
        )
        all_rows.append(row)

    # column names
    feature_columns = [f"f{i}" for i in range(features_vector.shape[0])]
    columns = feature_columns + metadata_keys

    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(output_csv, index=False)


def encode_metadata(metadata):
    encoded = []
    for k, v in metadata.items():
        if v is None or v == "indeterminate":
            encoded.append(None)
        elif (
            (k == "MGMT status" and v == "negative")
            or (k == "1p/19q" and v == "intact")
            or (k == "IDH" and v == "wildtype")
        ):
            encoded.append(0)
        else:
            encoded.append(1)
    return encoded


def extract_deep_features(image, mask, model, device=device):
    """
    Extracts both global features and mask-specific features from a 3D image.

    Args:
        image: numpy array [C, D, H, W]
        mask: numpy array [D, H, W] (0/1)
        model: pretrained UNet model
        device: 'cuda' or 'cpu'

    Returns:
        concatenated_features: torch tensor [1, 2*channels]
    """

    def extract_bottleneck_layer_features(image_tensor):
        bottleneck_features = []

        def hook_fn(module, input, output):
            bottleneck_features.append(output)

        # deepest encoder residual unit
        bottleneck_layer = model.bottleneck(model.model)
        handle = bottleneck_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            model(image_tensor)
        handle.remove()

        return bottleneck_features[0]  # tensor: [1, C, D, H, W]

    # Convert image and mask to torch tensors
    image_tensor = (
        torch.from_numpy(image).float().to(device).unsqueeze(0)
    )  # [1,C,D,H,W]
    mask_tensor = (
        torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
    )  # [1,1,D,H,W]

    # --- global features ---
    features = extract_bottleneck_layer_features(image_tensor)
    global_features = F.adaptive_avg_pool3d(features, (1, 1, 1)).view(
        features.size(0), -1
    )

    # --- masked features ---
    masked_image = image_tensor * mask_tensor  # zero-out outside mask
    features_masked = extract_bottleneck_layer_features(masked_image)
    segment_features = F.adaptive_avg_pool3d(features_masked, (1, 1, 1)).view(
        features.size(0), -1
    )

    # concatenate along channel dimension
    concatenated_features = torch.cat(
        [global_features, segment_features], dim=1
    )  # [1, 2*C]

    return concatenated_features


def combine_csv_rows(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    combined = pd.concat([df1, df2], axis=0, ignore_index=True)
    combined.to_csv(output_file, index=False)

    return combined


if __name__ == "__main__":
    main()
