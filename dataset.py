from config import dataset_json_paths, selected_metadata_keys
import csv
import json
import os
import nibabel as nib
import numpy as np
import re
import random

# Configure
MODALITIES = ["T1c", "T2", "FLAIR"]
IMAGE_EXT = ".nii.gz"
train_val_test_ratio = (0.8, 0.1, 0.1)


class Dataset:
    # --- Initialize --- #
    def __init__(self, images_dir, masks_dir, metadata_path, split="train"):
        """
        images_dir: path to images directory
        masks_dir: path to masks directory
        metadata: path to metadata csv file
        """

        split_to_idx = {"train": 0, "val": 1, "test": 2}
        json_path = dataset_json_paths[split_to_idx[split]]

        if not os.path.exists(json_path):
            self.split_dataset(images_dir, masks_dir)

        with open(json_path) as f:
            self.images_mask_paths = json.load(f)
        self.seek_idx = 0

    def validate_images_masks_paths(self, images_dir, masks_dir):
        images_mask_paths = []

        if not os.path.isdir(masks_dir):
            raise Exception("Masks folder does not exist!")

        for filename in os.listdir(masks_dir):
            base_name = self.mask_name_to_base(filename)
            mask_path = os.path.join(
                masks_dir, base_name + "_tumor_segmentation" + IMAGE_EXT
            )
            image_paths = []
            for modality in MODALITIES:
                image_name = os.path.join(
                    images_dir, base_name + "_" + modality + IMAGE_EXT
                )
                if os.path.exists(image_name):
                    image_paths.append(image_name)

            if os.path.isfile(mask_path) and len(image_paths) == 3:
                images_mask_paths.append((image_paths, mask_path))

        return images_mask_paths

    def mask_name_to_base(self, filename):
        match = re.match(r"([A-Z]+-[A-Z]+-\d+)_", filename)
        if match:
            result = match.group(1)
            return result

    # --- End of Initialize --- #

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.images_mask_paths)

    def __getitem__(self, idx):
        image, mask = self.id_to_image_mask(idx)
        return image, mask

    def __next__(self):
        if self.seek_idx >= len(self):
            raise StopIteration
        item = self.__getitem__(self.seek_idx)
        self.seek_idx += 1
        return item

    def id_to_image_mask(self, idx):
        """
        returns (image, mask) where image is (z, x, y, c) and mask is (z, x, y)
        """
        (image_paths, mask_path) = self.images_mask_paths[idx]

        image = []
        for image_path in image_paths:
            image.append(self.read_image(image_path))
        image = np.stack(image, axis=-1)

        mask = self.read_image(mask_path)

        image = np.astype(np.transpose(image, (3, 2, 0, 1)), np.float32)  # (C, D, H, W)
        mask = np.astype(np.transpose(mask, (2, 0, 1)), np.float32)  # (D, H, W)

        image = image[:, 13:-14, :, :]
        mask = mask[13:-14, :, :]

        return image, mask

    def read_image(self, image_path):
        return nib.load(image_path).get_fdata()

    def split_dataset(self, images_dir, masks_dir):
        # self.images_mask_paths: (((flair, t1c, t2), mask), ...)
        shuffled_paths = list(self.validate_images_masks_paths(images_dir, masks_dir))
        random.shuffle(shuffled_paths)
        shuffled_paths_length = len(shuffled_paths)

        paths = ([], [], [])

        for idx in range(len(train_val_test_ratio)):
            for _ in range(
                int(train_val_test_ratio[idx] * float(shuffled_paths_length))
            ):
                paths[idx].append(shuffled_paths.pop())

            json_output_path = dataset_json_paths[idx]
            if os.path.exists(json_output_path):
                raise Exception(
                    f"Split JSON already exists in {json_output_path}! Please remove it for the program to work correctly."
                )

            with open(json_output_path, "w") as f:
                json.dump(paths[idx], f)


class HybridDataset(Dataset):
    def __init__(self, images_dir, masks_dir, metadata_path, split=None):
        super().__init__(images_dir, masks_dir, metadata_path, split=split)
        self.metadata = self.read_csv_as_dicts(metadata_path)

    def __getitem__(self, idx):
        image, mask = self.id_to_image_mask(idx)
        _, mask_path = self.images_mask_paths[idx]
        patient_id = self.mask_name_to_base(os.path.basename(mask_path))
        for row in self.metadata:
            if row.get("ID") == patient_id:
                selected_metadata = {}
                for key in selected_metadata_keys:
                    selected_metadata.update({key: row.get(key)})
                return image, mask, selected_metadata

    def read_csv_as_dicts(self, file_path):
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            return [row for row in reader]

    def mask_name_to_base(self, filename):
        match = re.match(r"([A-Z]+-[A-Z]+-)(\d+)_", filename)
        if match:
            prefix, number = match.groups()
            number = str(int(number))
            return f"{prefix}{int(number):03d}"
