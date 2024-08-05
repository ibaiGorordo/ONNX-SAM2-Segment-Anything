import cv2
import numpy as np
import requests
from typing import Optional

rng = np.random.default_rng(2)
colors = rng.uniform(0, 255, size=(100, 3))


def draw_masks(
    image: np.ndarray,
    masks: list[np.ndarray],
    alpha: float = 0.5,
    draw_border: bool = True,
) -> np.ndarray:
    mask_image = image.copy()
    for label_id, label_masks in enumerate(masks):
        color = colors[label_id]
        if label_masks.ndim == 2:
            mask_image = draw_mask(
                mask_image,
                label_masks,
                (color[0], color[1], color[2]),
                alpha,
                draw_border,
            )
        else:
            for mask in label_masks:
                print(mask.shape)
                mask_image = draw_mask(
                    mask_image, mask, (color[0], color[1], color[2]), alpha, draw_border
                )

    return mask_image


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
    alpha: float = 0.5,
    draw_border: bool = True,
) -> np.ndarray:
    mask_image = image.copy()
    mask_image[mask > 0.01] = color
    mask_image = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)

    if draw_border:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(mask_image, contours, -1, color, thickness=2)

    return mask_image


def read_image_from_url(img_url: str) -> Optional[np.ndarray] :
    try:
        response = requests.get(img_url)

        if response.status_code == 200:
            img_data = response.content
            img_data = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.cvtColor(
                cv2.imdecode(img_data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            return img
        else:
            return None
    except Exception as e:
        raise FileNotFoundError({"error": repr(e)})
