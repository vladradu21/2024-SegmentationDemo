import time
from glob import glob
from operator import add

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from tqdm import tqdm

from model import build_unet
from utils import create_dir, seeding


def calculate_metrics(y_true, pred_y):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = pred_y.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_accuracy = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_accuracy]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  # (1024, 1024, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (1024, 1024, 3)
    return mask


if __name__ == '__main__':
    """ Seeding """
    seeding(42)

    """ Create folder """
    create_dir('../../out/results')

    """ Load dataset """
    test_x = glob("../../datasets/drishti-GS/test/*/image/*.png")
    test_y = glob("../../datasets/drishti-GS/test/*/mask/cup/*.png")

    """ Hyperparameters """
    H = 1024
    W = 1024
    size = (H, W)
    checkpoint_path = "files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = y.split('\\')[-1].split('.')[0]

        """ Read the images """
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # (1024, 1024, 3)
        x = np.transpose(image, (2, 0, 1))  # (3, 1024, 1024)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)  # (1, 3, 1024, 1024) batch of 1 image
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Read the masks """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # (1024, 1024)
        y = np.expand_dims(mask, axis=0)  # (1, 1024, 1024)
        y = y / 255.0
        y = np.expand_dims(y, axis=0)  # (1, 1, 1024, 1024)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            """ Prediction """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))

            pred_y = pred_y[0].cpu().numpy()  # (1, 1024, 1024)
            pred_y = np.squeeze(pred_y, axis=0)  # (1024, 1024)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Save mask """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate([image, line, ori_mask, line, pred_y * 255], axis=1)
        cv2.imwrite(f"../../out/results/{name}.png", cat_images)

    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)
