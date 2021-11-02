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


def create_dataset(dataset_path: str, train_size: float) -> None:
    labels = get_labels(dataset_path)

    splits = ["train", "val"]

    images = sorted([f for f in files(dataset_path)])
    assert len(images) == N_IMAGES

    for split in splits:
        for label in LABELS:
            label = label.replace(" ", "_")
            os.makedirs(os.path.join(dataset_path, split, label), exist_ok=True)

    n_train_images = int(train_size * IMAGES_PER_LABEL)

    for i, label in enumerate(LABELS):
        random.seed(label)
        
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="data/AFRIFASHION1600")
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--seed", type=float, default=23)
    args = parser.parse_args()

    random.seed(args.seed)
    create_dataset(args.dataset_path, args.train_size)
