import os
import random
import shutil
from typing import Dict

N_IMAGES = 1600
IMAGES_PER_LABEL = 200
LABELS = [
    "agbada",
    "blouse",
    "buba and trouser",
    "gele",
    "gown",
    "shirts", 
    "skirt and blouse",
    "wrapper and blouse"
]


def files(path: str):
    """
    https://stackoverflow.com/a/14176179/12160221
    """
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def get_labels(image_dir: str) -> Dict[str, str]:
    labels = {}
    for image in files(image_dir):
        labels[image] = (
            image.split("_")[1] if image.startswith("African_") else image.split("_")[0]
        ).lower().replace(" ", "_")
    
    return labels


def create_dataset(dataset_path: str, train_size: float, seeds: list) -> None:
    labels = get_labels(dataset_path)

    splits = ["train", "val"]

    images = sorted([f for f in files(dataset_path)])
    assert len(images) == N_IMAGES

    for split in splits:
        dir_name = os.path.join(dataset_path, split)
        os.makedirs(dir_name, exist_ok=True)

        for label in LABELS:
            label = label.replace(" ", "_")
            os.makedirs(os.path.join(dir_name, label), exist_ok=True)

    n_train_images = int(train_size * IMAGES_PER_LABEL)

    for i, seed in enumerate(LABELS):
        random.seed(seed)
        
        label_images = images[i*IMAGES_PER_LABEL:IMAGES_PER_LABEL * (i+1)]
        random.shuffle(label_images)

        for j, image in enumerate(label_images):
            shutil.move(
                os.path.join(dataset_path, image), 
                os.path.join(
                    dataset_path, "train" if j < n_train_images else "val", labels[image]
                )
            )


if __name__ == "__main__":
    create_dataset("data/AFRIFASHION1600", 0.7, [23, 54, 67, 98])