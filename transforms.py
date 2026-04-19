from monai.transforms import (
    Compose,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    EnsureTyped
)

train_transforms = Compose([
    # Intensity scaling
    ScaleIntensityRangePercentilesd(
        keys=["image"],
        lower=0.5,
        upper=99.5,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    # Normalization
    NormalizeIntensityd(
        keys=["image"],
        nonzero=True,
        channel_wise=True
    ),

    EnsureTyped(keys=["image", "mask"]),
])