import os
import numpy as np
import torch
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    compose_transforms,
    load_image_data,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    convert_to_tensor,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils import set_seed_for_determinism
from sklearn.model_selection import k_fold_cross_validator

set_seed_for_determinism(seed=42)

#paths to the dataset
data_dir = "./Task01_BrainTumour"
train_images_dir = os.path.join(data_dir, "imagesTr")
train_labels_dir = os.path.join(data_dir, "labelsTr")

#data preprocessing and augmentation
train_transforms = compose_transforms(
    [
        load_image_data(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        convert_to_tensor(keys=["image", "label"]),
    ]
)

# list of dictionaries
train_data = [
    {"image": os.path.join(train_images_dir, f), "label": os.path.join(train_labels_dir, f)}
    for f in os.listdir(train_images_dir) if f.endswith(".nii.gz")
]

train_ds = CacheDataset(data=train_data, transform=train_transforms)

# U-Net model
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# loss function and optimizer
dice_loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=True, reduction="mean")
hausdorff_metric = HausdorffDistanceMetric(include_background=True, percentile=95)

num_folds = 5
k_fold_cross_validator = k_fold_cross_validator(n_splits=num_folds, shuffle=True, random_state=42)

fold_dice_scores = []
fold_hausdorff_scores = []

for fold, (train_indices, validation_indices) in enumerate(k_fold_cross_validator.split(train_ds)):
    print(f"Fold [{fold+1}/{num_folds}]")

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_subsampler = torch.utils.data.SubsetRandomSampler(validation_indices)

    train_loader = DataLoader(train_ds, batch_size=2, sampler=train_subsampler, num_workers=4)
    val_loader = DataLoader(train_ds, batch_size=2, sampler=val_subsampler, num_workers=4)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            input_data, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            output_data = model(input_data)
            loss = dice_loss_function(output_data, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/step:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        dice_scores = {i: [] for i in range(4)}  
        hausdorff_scores = []
        for batch_data in val_loader:
            input_data, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            output_data = model(input_data)
            output_data_argmax = torch.argmax(output_data, dim=1)
            labels = labels.squeeze(dim=1)
            
            for i in range(4):
                dice_score_class = dice_metric(output_data_argmax == i, labels == i)
                dice_scores[i].append(dice_score_class.mean().item())
            
            hausdorff_score = hausdorff_metric(output_data_argmax, labels)
            hausdorff_scores.append(hausdorff_score.mean().item())
        
        # Dice scores for each class
        avg_dice_scores = {i: np.mean(scores) for i, scores in dice_scores.items()}
        fold_dice_scores.append(avg_dice_scores)
        
        fold_hausdorff_scores.append(np.mean(hausdorff_scores))
        
        print(f"Fold [{fold+1}/{num_folds}] Dice Scores:")
        for i, score in avg_dice_scores.items():
            print(f"  Class {i}: {score:.4f}")
        print(f"Fold [{fold+1}/{num_folds}] Hausdorff Distance: {np.mean(hausdorff_scores):.4f}")

# Printing results
print("Average Dice Scores:")
for i in range(4):
    class_dice_scores = [fold_scores[i] for fold_scores in fold_dice_scores]
    avg_dice_score = np.mean(class_dice_scores)
    print(f"  Class {i}: {avg_dice_score:.4f}")

print(f"Average Hausdorff Distance: {np.mean(fold_hausdorff_scores):.4f}")