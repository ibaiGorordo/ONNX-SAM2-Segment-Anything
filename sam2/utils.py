import cv2
import numpy as np

rng = np.random.default_rng(2)
colors = rng.uniform(0, 255, size=(100, 3))

def draw_masks(image: np.ndarray, masks: dict[int, np.ndarray], alpha: float = 0.5, draw_border: bool = True) -> np.ndarray:
    mask_image = image.copy()

    for label_id, label_masks in masks.items():
        if label_masks is None:
            continue
        color = colors[label_id]
        mask_image = draw_mask(mask_image, label_masks, (color[0], color[1], color[2]), alpha, draw_border)

    return mask_image

def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.5, draw_border: bool = True) -> np.ndarray:
    mask_image = image.copy()
    mask_image[mask > 0.01] = color
    mask_image = cv2.addWeighted(image, 1-alpha, mask_image, alpha, 0)

    if draw_border:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(mask_image, contours, -1, color, thickness=2)

    return mask_image
