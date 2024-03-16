import os
from glob import glob
from pathlib import Path

import cv2
import imageio
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SIZE = (1024, 1024)


def create_directories(data_dir):
    for subdir in ["train/normal/image", "train/glaucoma/image",
                   "train/normal/mask/cup", "train/glaucoma/mask/cup",
                   "train/normal/mask/disc", "train/glaucoma/mask/disc",
                   "test/normal/image", "test/glaucoma/image",
                   "test/normal/mask/cup", "test/glaucoma/mask/cup",
                   "test/normal/mask/disc", "test/glaucoma/mask/disc"]:
        (Path(data_dir) / subdir).mkdir(parents=True, exist_ok=True)


def load_rim_one_data(path):
    h_images = sorted(glob(os.path.join(path, "healthy", "stereo images", "*.jpg")))
    h_cup_masks = sorted(glob(os.path.join(path, "healthy", "average_masks", "*Cup*.png")))
    h_disc_masks = sorted(glob(os.path.join(path, "healthy", "average_masks", "*Disc*.png")))

    g_images = sorted(glob(os.path.join(path, "Glaucoma and suspects", "stereo images", "*.jpg")))
    g_cup_masks = sorted(glob(os.path.join(path, "Glaucoma and suspects", "average_masks", "*Cup*.png")))
    g_disc_masks = sorted(glob(os.path.join(path, "Glaucoma and suspects", "average_masks", "*Disc*.png")))

    return ((h_images, h_cup_masks, h_disc_masks),
            (g_images, g_cup_masks, g_disc_masks))


def load_drishti_gs_data(path):
    h_images = sorted(glob(os.path.join(path, "normal", "images", "*.png")))
    h_cup_masks = sorted(glob(os.path.join(path, "normal", "GT", "*", "SoftMap", "*cup*.png")))
    h_disc_masks = sorted(glob(os.path.join(path, "normal", "GT", "*", "SoftMap", "*OD*.png")))

    g_images = sorted(glob(os.path.join(path, "glaucoma", "images", "*.png")))
    g_cup_masks = sorted(glob(os.path.join(path, "glaucoma", "GT", "*", "SoftMap", "*cup*.png")))
    g_disc_masks = sorted(glob(os.path.join(path, "glaucoma", "GT", "*", "SoftMap", "*OD*.png")))

    return ((h_images, h_cup_masks, h_disc_masks),
            (g_images, g_cup_masks, g_disc_masks))


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
    create_directories(directory_path)

    ((train_images, train_cup_masks, train_disc_masks),
     (test_images, test_cup_masks, test_disc_masks)) = split_data(images, cup_masks, disc_masks)
    print(f"Train images: {len(train_images)} - Test images: {len(test_images)} for {directory_path} {folder}")

    resize_data(train_images, train_cup_masks, train_disc_masks, str(directory_path + "/train/" + folder))
    resize_data(test_images, test_cup_masks, test_disc_masks, str(directory_path + "/test/" + folder))


def load_data():
    """ Load the data RIM ONE """
    rim_one_data_path = r"D:\licenta\datasets\RIM-ONE r3 - Copy"
    ((healthy_images, healthy_cup_masks, healthy_disc_masks),
     (glaucoma_images, glaucoma_cup_masks, glaucoma_disc_masks)) = load_rim_one_data(rim_one_data_path)

    handle_data(healthy_images, healthy_cup_masks, healthy_disc_masks, "../data/rim_one_r3", "normal")
    handle_data(glaucoma_images, glaucoma_cup_masks, glaucoma_disc_masks, "../data/rim_one_r3", "glaucoma")

    """ Load the data DRISHTI """
    drishti_data_path = r"D:\licenta\datasets\Drishti-GS - Copy"
    ((healthy_images, healthy_cup_masks, healthy_disc_masks),
     (glaucoma_images, glaucoma_cup_masks, glaucoma_disc_masks)) = load_drishti_gs_data(drishti_data_path)

    handle_data(healthy_images, healthy_cup_masks, healthy_disc_masks, "../data/drishti-GS", "normal")
    handle_data(glaucoma_images, glaucoma_cup_masks, glaucoma_disc_masks, "../data/drishti-GS", "glaucoma")


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    load_data()
