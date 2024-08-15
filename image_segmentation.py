from sam2 import SAM2Image, draw_masks
import cv2
import numpy as np
from imread_from_url import imread_from_url


encoder_model_path = "models/sam2_hiera_base_plus_encoder.onnx"
decoder_model_path = "models/sam2_hiera_base_plus_decoder.onnx"

img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Racing_Terriers_%282490056817%29.jpg/1280px-Racing_Terriers_%282490056817%29.jpg"
img = imread_from_url(img_url)

# Initialize models
sam2 = SAM2Image(encoder_model_path, decoder_model_path)

# Set image
sam2.set_image(img)

# Add points
point_coords = [np.array([[420, 440]]), np.array([[360, 275], [370, 210]]), np.array([[810, 440]]),
                np.array([[920, 314]])]
point_labels = [np.array([1]), np.array([1, 1]), np.array([1]), np.array([1])]

for label_id, (point_coord, point_label) in enumerate(zip(point_coords, point_labels)):
    for i in range(point_label.shape[0]):
        sam2.add_point((point_coord[i][0], point_coord[i][1]), point_label[i], label_id)

    # Decode image
    masks = sam2.update_mask()

    # Draw masks
    masked_img = draw_masks(img, masks)

    cv2.imshow("masked_img", masked_img)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
