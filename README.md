# MRI Biomarker Prediction

## Structure
* **Dataset**: [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/)
* **Model**:
    - Deep feature extraction: PyTorch UNet 3D, bottleneck layer
    - XGboost trained on deep features -> biomarker prediction

## Usage
* **Clone** this repository
* **Download** the dataset and copy files into `dataset/`. In masks folder, we only put whole tumor segmentation masks.
* Make sure the **file structure** is according to this:
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
* Train the deep learning model
```
python train.py
```
* Extract features and export to CSV
```
python extract_features.py
```
* Train XGBoost models using the exported CSV
```
python train_xgboost.py
```