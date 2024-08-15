import time
from typing import Any

import cv2
import numpy as np
import onnxruntime
from numpy import ndarray


class SAM2Image:
    def __init__(self, encoder_path: str, decoder_path: str) -> None:
        # Initialize models
        self.encoder = SAM2ImageEncoder(encoder_path)
        self.decoder = SAM2ImageDecoder(decoder_path, self.encoder.input_shape[2:])

        self.point_coords = []
        self.point_labels = []

    def set_image(self, image: np.ndarray) -> None:
        self.image_embeddings = self.encoder(image)
        self.orig_im_size = (image.shape[0], image.shape[1])
        self.decoder.set_image_size((image.shape[0], image.shape[1]))
        self.reset_points()

    def update_mask(self) -> list[np.ndarray]:
        if not self.point_coords:
            return [np.empty(0)]

        high_res_feats_0, high_res_feats_1, image_embed = self.image_embeddings
        masks, _ = self.decoder(image_embed, high_res_feats_0, high_res_feats_1, self.point_coords, self.point_labels)

        # Set the mask to zeros if no points are present
        for i, mask in enumerate(masks):
            if self.point_coords[i].shape[0] == 0:
                masks[i] = np.zeros((self.orig_im_size[0], self.orig_im_size[1]), dtype=np.uint8)

        return masks

    def add_point(self, point_coords: tuple[int, int], is_positive: bool, label_id: int) -> None:
        if label_id >= len(self.point_coords):
            self.point_coords.append(np.array([point_coords]))
            self.point_labels.append(np.array([1 if is_positive else 0]))
        else:
            # new_point_coords = np.array([point_coords])
            self.point_coords[label_id] = np.append(self.point_coords[label_id], np.array([point_coords]), axis=0)
            self.point_labels[label_id] = np.append(self.point_labels[label_id], 1 if is_positive else 0)

    def remove_point(self,  point_coords: tuple[int, int], label_id: int) -> None:
        point_id = np.where((self.point_coords[label_id][:, 0] == point_coords[0]) & (self.point_coords[label_id][:, 1] == point_coords[1]))[0][0]
        self.point_coords[label_id] = np.delete(self.point_coords[label_id], point_id, axis=0)
        self.point_labels[label_id] = np.delete(self.point_labels[label_id], point_id, axis=0)

    def reset_points(self) -> None:
        self.point_coords = []
        self.point_labels = []


class SAM2ImageEncoder:
    def __init__(self, path: str) -> None:
        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.encode_image(image)

    def encode_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.prepare_input(image)

        outputs = self.infer(input_tensor)

        return self.process_output(outputs)

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = (input_img / 255.0 - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return outputs[0], outputs[1], outputs[2]

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


class SAM2ImageDecoder:
    def __init__(self, path: str,
                 encoder_input_size: tuple[int, int],
                 orig_im_size: tuple[int, int] = None,
                 mask_threshold: float = 0.0) -> None:
        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())

        self.orig_im_size = orig_im_size if orig_im_size is not None else encoder_input_size
        self.encoder_input_size = encoder_input_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image_embed: np.ndarray,
                 high_res_feats_0: np.ndarray, high_res_feats_1: np.ndarray,
                 point_coords: list[np.ndarray] | np.ndarray,
                 point_labels: list[np.ndarray] | np.ndarray) -> tuple[list[np.ndarray], ndarray]:

        return self.predict(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels)

    def predict(self, image_embed: np.ndarray,
                 high_res_feats_0: np.ndarray, high_res_feats_1: np.ndarray,
                 point_coords: list[np.ndarray] | np.ndarray,
                 point_labels: list[np.ndarray] | np.ndarray) -> tuple[list[np.ndarray], ndarray]:

        inputs = self.prepare_inputs(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels)

        outputs = self.infer(inputs)

        return self.process_output(outputs)


    def prepare_inputs(self, image_embed: np.ndarray,
                       high_res_feats_0: np.ndarray, high_res_feats_1: np.ndarray,
                       point_coords: list[np.ndarray] | np.ndarray,
                       point_labels: list[np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        input_point_coords, input_point_labels = self.prepare_points(point_coords, point_labels)

        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros((num_labels, 1, self.encoder_input_size[0] // self.scale_factor, self.encoder_input_size[1] // self.scale_factor), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        original_size = np.array([self.orig_im_size[0], self.orig_im_size[1]], dtype=np.int32)

        return image_embed, high_res_feats_0, high_res_feats_1, input_point_coords, input_point_labels, mask_input, has_mask_input, original_size


    def prepare_points(self, point_coords: list[np.ndarray] | np.ndarray, point_labels: list[np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        if isinstance(point_coords, np.ndarray):
            input_point_coords = point_coords[np.newaxis, ...]
            input_point_labels = point_labels[np.newaxis, ...]
        else:
            max_num_points = max([coords.shape[0] for coords in point_coords])
            # We need to make sure that all inputs have the same number of points
            # Add invalid points to pad the input (0, 0) with -1 value for labels
            input_point_coords = np.zeros((len(point_coords), max_num_points, 2), dtype=np.float32)
            input_point_labels = np.ones((len(point_coords), max_num_points), dtype=np.float32) * -1

            for i, (coords, labels) in enumerate(zip(point_coords, point_labels)):
                input_point_coords[i, :coords.shape[0], :] = coords
                input_point_labels[i, :labels.shape[0]] = labels

        input_point_coords[..., 0] = input_point_coords[..., 0] / self.orig_im_size[1] * self.encoder_input_size[1]  # Normalize x
        input_point_coords[..., 1] = input_point_coords[..., 1] / self.orig_im_size[0] * self.encoder_input_size[0]  # Normalize y

        return input_point_coords.astype(np.float32), input_point_labels.astype(np.float32)

    def infer(self, inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> list[np.ndarray]:
        start = time.perf_counter()

        outputs = self.session.run(self.output_names,
                                   {self.input_names[i]: inputs[i] for i in range(len(self.input_names))})

        print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, outputs: list[np.ndarray]) -> tuple[list[ndarray | Any], ndarray[Any, Any]]:

        scores = outputs[1].squeeze()
        masks = outputs[0]
        masks = masks > self.mask_threshold
        masks = masks.astype(np.uint8)

        output_masks = []
        for label_id in range(masks.shape[0]):
            label_masks = masks[label_id, 0, ...]
            output_masks.append(label_masks)

        return output_masks, scores

    def set_image_size(self, orig_im_size: tuple[int, int]) -> None:
        self.orig_im_size = orig_im_size

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from utils import draw_masks
    from imread_from_url import imread_from_url

    encoder_model_path = "../models/sam2_hiera_base_plus_encoder.onnx"
    decoder_model_path = "../models/sam2_hiera_base_plus_decoder.onnx"

    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Racing_Terriers_%282490056817%29.jpg/1280px-Racing_Terriers_%282490056817%29.jpg"
    img = imread_from_url(img_url)

    # Initialize models
    sam2_encoder = SAM2ImageEncoder(encoder_model_path)
    sam2_decoder = SAM2ImageDecoder(decoder_model_path, sam2_encoder.input_shape[2:], img.shape[:2])

    # Encode image
    high_res_feats_0, high_res_feats_1, image_embed = sam2_encoder(img)

    point_coords = [np.array([[420,440]]), np.array([[360, 275], [370, 210]]), np.array([[810, 440]]), np.array([[920, 314]])]
    point_labels = [np.array([1]), np.array([1,1]), np.array([1]), np.array([1])]

    # Decode image
    masks, scores = sam2_decoder(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels)

    masked_img = draw_masks(img, masks)

    cv2.imwrite("../doc/img/sam2_masked_img.jpg", masked_img)

    cv2.imshow("masked_img", masked_img)
    cv2.waitKey(0)