import os

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


def create_labels(image_dir: str, output_dir: str = None):
    for image in os.listdir(image_dir):
        image_type = (
            image.split("_")[1] if image.startswith("African_") else image.split("_")[0]
        ).lower()
        print(image_type)
    
create_labels("data/AFRIFASHION1600")