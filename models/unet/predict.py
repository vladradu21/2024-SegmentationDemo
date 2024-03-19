import os

import cv2
import numpy as np
import torch

from model import build_unet


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask


if __name__ == '__main__':
    """ Load the image """
    image_path = '../../data/predict/image/fundus.png'
    image_name = os.path.splitext(os.path.basename(image_path))[0]  # base name of the file without extension
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 512))  # Resize if necessary
    x = np.transpose(image, (2, 0, 1))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)

    """ Set device """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ Load the model """
    checkpoint_path = 'files/checkpoint.pth'
    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Prediction """
    x = x.to(device)
    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

    """ Save or display the prediction """
    pred_mask = mask_parse(pred_y)
    save_path = f'../../data/predict/predicted/cup/{image_name}_cup.png'
    cv2.imwrite(save_path, pred_mask * 255)

    print(f"Saved predicted mask to {save_path}")
