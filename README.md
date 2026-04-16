# MRI Biomarker Prediction

## Structure
* **Dataset**: [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/)
* **Model**:
    - Deep feature extraction: PyTorch UNet 3D, bottleneck layer
    - XGboost trained on deep features -> biomarker prediction

## Usage
* **Clone** this repository
* **Download** the dataset and move it into the project root of this repository.
* **Prepare dataset** using the next command:
```
bash prepare_dataset.sh
```
* After doing the previous step, make sure the **file structure** is according to this:
```md
    - dataset/
        - images/
            - UCSF-PDGM-0023_FLAIR.nii.gz
            - ...
        - masks/
            - UCSF-PDGM-0023_tumor_segmentation_whole_tumor.nii.gz
            - ...
        - UCSF-PDGM-metadata_v2.csv
```

* **Train the deep learning model**
```
python train.py
```

* **Move model checkpoint**: After training the deep learning model, you will find checkpoints under `lightning_logs/version_0/checkpoints`. Choose one based on lowest loss, copy it to main project folder and name it `checkpoint.ckpt` so `extract_features.py` will be able to find it.

* **Extract features and export to CSV**
```
python extract_features.py
```

* **Train XGBoost models using the exported CSV**
```
python train_xgboost.py
```