import os
from glob import glob
from pathlib import Path

import cv2
import imageio
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SIZE = (1024, 1024)

dataset_configurations = {
    "RIM_ONE": {
        "healthy": {
            "images": "healthy/stereo images/*.jpg",
            "cup_masks": "healthy/average_masks/*Cup*.png",
            "disc_masks": "healthy/average_masks/*Disc*.png"
        },
        "glaucoma": {
            "images": "Glaucoma and suspects/stereo images/*.jpg",
            "cup_masks": "Glaucoma and suspects/average_masks/*Cup*.png",
            "disc_masks": "Glaucoma and suspects/average_masks/*Disc*.png"
        }
    },
    "DRISHTI_GS": {
        "healthy": {
            "images": "normal/images/*.png",
            "cup_masks": "normal/GT/*/SoftMap/*cup*.png",
            "disc_masks": "normal/GT/*/SoftMap/*OD*.png"
        },
        "glaucoma": {
            "images": "glaucoma/images/*.png",
            "cup_masks": "glaucoma/GT/*/SoftMap/*cup*.png",
            "disc_masks": "glaucoma/GT/*/SoftMap/*OD*.png"
        }
    }
}


def create_directories(data_dir):
    categories = ['normal', 'glaucoma']
    types = ['image', 'mask/cup', 'mask/disc']
    phases = ['train', 'test']

    for phase in phases:
        for category in categories:
            for dtype in types:
                subdir = f"{phase}/{category}/{dtype}"
                (Path(data_dir) / subdir).mkdir(parents=True, exist_ok=True)


def load_data(path, config):
    results = []
    for condition in ['healthy', 'glaucoma']:
        paths = config[condition]
        images = sorted(glob(os.path.join(path, paths["images"])))
        cup_masks = sorted(glob(os.path.join(path, paths["cup_masks"])))
        disc_masks = sorted(glob(os.path.join(path, paths["disc_masks"])))
        results.append((images, cup_masks, disc_masks))
    return tuple(results)


def resize_data(images, cup_masks, disc_masks, save_path):
    for idx, (x, y, z) in tqdm(enumerate(zip(images, cup_masks, disc_masks)), total=len(images)):
        name = x.split("\\")[-1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]
        z = imageio.mimread(z)[0]

        # Resize images
        x_resized = cv2.resize(x, SIZE)
        y_resized = cv2.resize(y, SIZE, interpolation=cv2.INTER_NEAREST)
        z_resized = cv2.resize(z, SIZE, interpolation=cv2.INTER_NEAREST)

        # Prepare save paths
        image_path = os.path.join(save_path, "image", f"{name}_image.png")
        cup_mask_path = os.path.join(save_path, "mask", "cup", f"{name}_cup_mask.png")
        disc_mask_path = os.path.join(save_path, "mask", "disc", f"{name}_disc_mask.png")

        # Save resized images and masks
        cv2.imwrite(image_path, x_resized)
        cv2.imwrite(cup_mask_path, y_resized)
        cv2.imwrite(disc_mask_path, z_resized)


def split_data(images, cup_masks, disc_masks, test_size=0.2, random_state=None):
    train_images, test_images, train_cup_masks, test_cup_masks, train_disc_masks, test_disc_masks = train_test_split(
        images, cup_masks, disc_masks, test_size=test_size, random_state=random_state, shuffle=True)

    return (train_images, train_cup_masks, train_disc_masks), (test_images, test_cup_masks, test_disc_masks)


def handle_data(images, cup_masks, disc_masks, directory_path, folder):
    ((train_images, train_cup_masks, train_disc_masks),
     (test_images, test_cup_masks, test_disc_masks)) = split_data(images, cup_masks, disc_masks)
    print(f"Train images: {len(train_images)} - Test images: {len(test_images)} for {directory_path} {folder}")

    resize_data(train_images, train_cup_masks, train_disc_masks, str(directory_path + "/train/" + folder))
    resize_data(test_images, test_cup_masks, test_disc_masks, str(directory_path + "/test/" + folder))


def define_dataset():
    datasets_info = [
        {
            "data_path": r"D:\licenta\datasets\RIM-ONE r3 - Copy",
            "config": dataset_configurations["RIM_ONE"],
            "output_dir": "../data/rim-one-r3"
        },
        {
            "data_path": r"D:\licenta\datasets\Drishti-GS - Copy",
            "config": dataset_configurations["DRISHTI_GS"],
            "output_dir": "../data/drishti-GS"
        }
    ]

    for dataset in datasets_info:
        data_path, config, output_dir = dataset.values()
        (hi, hc, hd), (gi, gc, gd) = load_data(data_path, config)

        create_directories(output_dir)
        handle_data(hi, hc, hd, output_dir, "normal")
        handle_data(gi, gc, gd, output_dir, "glaucoma")


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    define_dataset()
