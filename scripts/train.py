from pandas_path import path
from pathlib import Path
import numpy as np
import pandas as pd
import random

import albumentations
import pytorch_lightning as pl
from train_src.cloud_model import CloudModel

# import warnings
# warnings.filterwarnings("ignore")

# DATA DIRECTORY
DATA_DIR = Path("/driven-data/cloud-cover")
TRAIN_FEATURES = DATA_DIR / "train_features"
TRAIN_LABELS = DATA_DIR / "train_labels"

# Confirm the folder exists.
assert TRAIN_FEATURES.exists()

# BANDS
BANDS = ["B02", "B03", "B04", "B08"]


# MAIN
def main():
    # Read in the train meta.
    train_meta = pd.read_csv(DATA_DIR / "train_metadata.csv")
    train_meta = add_paths(train_meta, TRAIN_FEATURES, TRAIN_LABELS)

    # Train/test split
    random.seed(9)  # set a seed for reproducibility

    # put 1/3 of chips into the validation set
    chip_ids = train_meta.chip_id.unique().tolist()
    val_chip_ids = random.sample(chip_ids, round(len(chip_ids) * 0.33))

    val_mask = train_meta.chip_id.isin(val_chip_ids)
    val = train_meta[val_mask].copy().reset_index(drop=True)
    train = train_meta[~val_mask].copy().reset_index(drop=True)

    # separate features from labels
    feature_cols = ["chip_id"] + [f"{band}_path" for band in BANDS]

    val_x = val[feature_cols].copy()
    val_y = val[["chip_id", "label_path"]].copy()

    train_x = train[feature_cols].copy()
    train_y = train[["chip_id", "label_path"]].copy()
    
    # Data transforms.
    training_transformations = None
    
    # Set up pytorch_lightning.Trainer object
    cloud_model = CloudModel(
        bands=BANDS,
        x_train=train_x,
        y_train=train_y,
        x_val=val_x,
        y_val=val_y,
        hparams={
            "num_workers": 4, 
            "batch_size": 8,
            # "transforms": training_transformations
        },
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="iou_epoch", mode="max", verbose=True,
    )
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="iou_epoch",
        patience=(cloud_model.patience * 3),
        mode="max",
        verbose=True,
    )

    trainer = pl.Trainer(
        gpus=1,
        fast_dev_run=False,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # Fit the model
    # This will initialize on Version 14
    trainer.fit(model=cloud_model, ckpt_path='lightning_logs/version_13/checkpoints/epoch=12-step=12791.ckpt')

# UTILITIES
def add_paths(df, feature_dir, label_dir=None, bands=BANDS):
    """
    Given dataframe with a column for chip_id, returns a dataframe with a column
    added indicating the path to each band's TIF image as "{band}_path", eg "B02_path".
    A column is also added to the dataframe with paths to the label TIF, if the
    path to the labels directory is provided.
    """
    for band in bands:
        df[f"{band}_path"] = feature_dir / df["chip_id"] / f"{band}.tif"
        # make sure a random sample of paths exist
        assert df.sample(n=40, random_state=5)[f"{band}_path"].path.exists().all()
    if label_dir is not None:
        df["label_path"] = label_dir / (df["chip_id"] + ".tif")
        # make sure a random sample of paths exist
        assert df.sample(n=40, random_state=5)["label_path"].path.exists().all()
    return df


if __name__ == '__main__':
    main()