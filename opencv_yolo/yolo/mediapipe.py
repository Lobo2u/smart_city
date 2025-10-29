import cv2 as cv
import numpy as np


class MPHandPose:
    def __init__(self, modelPath, confThreshold=0.8, backendId=0, targetId=0):
        self.model_path = modelPath
        self.conf_threshold = confThreshold
        self.backend_id = backendId
        self.target_id = targetId

        self.input_size = np.array([224, 224])  # wh
        self.PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2]
        self.PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0
        self.PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2
        self.PALM_BOX_PRE_SHIFT_VECTOR = [0, 0]
        self.PALM_BOX_PRE_ENLARGE_FACTOR = 4
        self.PALM_BOX_SHIFT_VECTOR = [0, -0.4]
        self.PALM_BOX_ENLARGE_FACTOR = 3
        self.HAND_BOX_SHIFT_VECTOR = [0, -0.1]
        self.HAND_BOX_ENLARGE_FACTOR = 1.65

        self.model = cv.dnn.readNet(self.model_path)
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self.backend_id = backendId
        self.target_id = targetId
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

    def _cropAndPadFromPalm(self, image, palm_bbox, for_rotation = False):
        # shift bounding box
        wh_palm_bbox = palm_bbox[1] - palm_bbox[0]
        if for_rotation:
            shift_vector = self.PALM_BOX_PRE_SHIFT_VECTOR
        else:
            shift_vector = self.PALM_BOX_SHIFT_VECTOR
        shift_vector = shift_vector * wh_palm_bbox
        palm_bbox = palm_bbox + shift_vector
        # enlarge bounding box
        center_palm_bbox = np.sum(palm_bbox, axis=0) / 2
        wh_palm_bbox = palm_bbox[1] - palm_bbox[0]
        if for_rotation:
            enlarge_scale = self.PALM_BOX_PRE_ENLARGE_FACTOR
        else:
            enlarge_scale = self.PALM_BOX_ENLARGE_FACTOR
        new_half_size = wh_palm_bbox * enlarge_scale / 2
        palm_bbox = np.array([
            center_palm_bbox - new_half_size,
            center_palm_bbox + new_half_size])
        palm_bbox = palm_bbox.astype(np.int32)
        palm_bbox[:, 0] = np.clip(palm_bbox[:, 0], 0, image.shape[1])
        palm_bbox[:, 1] = np.clip(palm_bbox[:, 1], 0, image.shape[0])
        # crop to the size of interest
        image = image[palm_bbox[0][1]:palm_bbox[1][1], palm_bbox[0][0]:palm_bbox[1][0], :]
        # pad to ensure conner pixels won't be cropped
        if for_rotation:
            side_len = np.linalg.norm(image.shape[:2])
        else:
            side_len = max(image.shape[:2])

        side_len = int(side_len)
        pad_h = side_len - image.shape[0]
        pad_w = side_len - image.shape[1]
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))
        bias = palm_bbox[0] - [left, top]
        return image, palm_bbox, bias

    def _preprocess(self, image, palm):
        '''
        Rotate input for inference.
        Parameters:
          image - input image of BGR channel order
          palm_bbox - palm bounding box found in image of format [[x1, y1], [x2, y2]] (top-left and bottom-right points)
          palm_landmarks - 7 landmarks (5 finger base points, 2 palm base points) of shape [7, 2]
        Returns:
          rotated_hand - rotated hand image for inference
          rotate_palm_bbox - palm box of interest range
          angle - rotate angle for hand
          rotation_matrix - matrix for rotation and de-rotation
          pad_bias - pad pixels of interest range
        '''
        # crop and pad image to interest range
        pad_bias = np.array([0, 0], dtype=np.int32)  # left, top
        palm_bbox = palm[0:4].reshape(2, 2)
        image, palm_bbox, bias = self._cropAndPadFromPalm(image, palm_bbox, True)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        pad_bias += bias

        # Rotate input to have vertically oriented hand image
        # compute rotation
        palm_bbox -= pad_bias
        palm_landmarks = palm[4:18].reshape(7, 2) - pad_bias
        p1 = palm_landmarks[self.PALM_LANDMARKS_INDEX_OF_PALM_BASE]
        p2 = palm_landmarks[self.PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE]
        radians = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        radians = radians - 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = np.rad2deg(radians)
        #  get bbox center
        center_palm_bbox = np.sum(palm_bbox, axis=0) / 2
        #  get rotation matrix
        rotation_matrix = cv.getRotationMatrix2D(center_palm_bbox, angle, 1.0)
        #  get rotated image
        rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        #  get bounding boxes from rotated palm landmarks
        homogeneous_coord = np.c_[palm_landmarks, np.ones(palm_landmarks.shape[0])]
        rotated_palm_landmarks = np.array([
            np.dot(homogeneous_coord, rotation_matrix[0]),
            np.dot(homogeneous_coord, rotation_matrix[1])])
        #  get landmark bounding box
        rotated_palm_bbox = np.array([
            np.amin(rotated_palm_landmarks, axis=1),
            np.amax(rotated_palm_landmarks, axis=1)])  # [top-left, bottom-right]

        crop, rotated_palm_bbox, _ = self._cropAndPadFromPalm(rotated_image, rotated_palm_bbox)
        blob = cv.resize(crop, dsize=self.input_size, interpolation=cv.INTER_AREA).astype(np.float32)
        blob = blob / 255.

        return blob[np.newaxis, :, :, :], rotated_palm_bbox, angle, rotation_matrix, pad_bias

    def infer(self, image, palm):
        # Preprocess
        input_blob, rotated_palm_bbox, angle, rotation_matrix, pad_bias = self._preprocess(image, palm)

        # Forward
        self.model.setInput(input_blob)
        output_blob = self.model.forward(self.model.getUnconnectedOutLayersNames())

        # Postprocess
        results = self._postprocess(output_blob, rotated_palm_bbox, angle, rotation_matrix, pad_bias)
        return results # [bbox_coords, landmarks_coords, conf]

    def _postprocess(self, blob, rotated_palm_bbox, angle, rotation_matrix, pad_bias):
        landmarks, conf, handedness, landmarks_word = blob

        conf = conf[0][0]
        if conf < self.conf_threshold:
            return None

        landmarks = landmarks[0].reshape(-1, 3)  # shape: (1, 63) -> (21, 3)
        landmarks_word = landmarks_word[0].reshape(-1, 3) # shape: (1, 63) -> (21, 3)

        # transform coords back to the input coords
        wh_rotated_palm_bbox = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        scale_factor = wh_rotated_palm_bbox / self.input_size
        landmarks[:, :2] = (landmarks[:, :2] - self.input_size / 2) * max(scale_factor)
        landmarks[:, 2] = landmarks[:, 2] * max(scale_factor) # depth scaling
        coords_rotation_matrix = cv.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated_landmarks = np.dot(landmarks[:, :2], coords_rotation_matrix[:, :2])
        rotated_landmarks = np.c_[rotated_landmarks, landmarks[:, 2]]
        rotated_landmarks_world = np.dot(landmarks_word[:, :2], coords_rotation_matrix[:, :2])
        rotated_landmarks_world = np.c_[rotated_landmarks_world, landmarks_word[:, 2]]
        #  invert rotation
        rotation_component = np.array([
            [rotation_matrix[0][0], rotation_matrix[1][0]],
            [rotation_matrix[0][1], rotation_matrix[1][1]]])
        translation_component = np.array([
            rotation_matrix[0][2], rotation_matrix[1][2]])
        inverted_translation = np.array([
            -np.dot(rotation_component[0], translation_component),
            -np.dot(rotation_component[1], translation_component)])
        inverse_rotation_matrix = np.c_[rotation_component, inverted_translation]
        #  get box center
        center = np.append(np.sum(rotated_palm_bbox, axis=0) / 2, 1)
        original_center = np.array([
            np.dot(center, inverse_rotation_matrix[0]),
            np.dot(center, inverse_rotation_matrix[1])])
        landmarks[:, :2] = rotated_landmarks[:, :2] + original_center + pad_bias

        # get bounding box from rotated_landmarks
        bbox = np.array([
            np.amin(landmarks[:, :2], axis=0),
            np.amax(landmarks[:, :2], axis=0)])  # [top-left, bottom-right]
        # shift bounding box
        wh_bbox = bbox[1] - bbox[0]
        shift_vector = self.HAND_BOX_SHIFT_VECTOR * wh_bbox
        bbox = bbox + shift_vector
        # enlarge bounding box
        center_bbox = np.sum(bbox, axis=0) / 2
        wh_bbox = bbox[1] - bbox[0]
        new_half_size = wh_bbox * self.HAND_BOX_ENLARGE_FACTOR / 2
        bbox = np.array([
            center_bbox - new_half_size,
            center_bbox + new_half_size])

        # [0: 4]: hand bounding box found in image of format [x1, y1, x2, y2] (top-left and bottom-right points)
        # [4: 67]: screen landmarks with format [x1, y1, z1, x2, y2 ... x21, y21, z21], z value is relative to WRIST
        # [67: 130]: world landmarks with format [x1, y1, z1, x2, y2 ... x21, y21, z21], 3D metric x, y, z coordinate
        # [130]: handedness, (left)[0, 1](right) hand
        # [131]: confidence
        return np.r_[bbox.reshape(-1), landmarks.reshape(-1), rotated_landmarks_world.reshape(-1), handedness[0][0], conf]

class MPPalmDet:
    def __init__(self, modelPath, nmsThreshold=0.3, scoreThreshold=0.5, topK=5000, backendId=0, targetId=0):
        self.model_path = modelPath
        self.nms_threshold = nmsThreshold
        self.score_threshold = scoreThreshold
        self.topK = topK
        self.backend_id = backendId
        self.target_id = targetId

        self.input_size = np.array([192, 192]) # wh

        self.model = cv.dnn.readNet(self.model_path)
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

        self.anchors = self._load_anchors()

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self.backend_id = backendId
        self.target_id = targetId
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

    def _preprocess(self, image):
        pad_bias = np.array([0., 0.]) # left, top
        ratio = min(self.input_size / image.shape[:2])
        if image.shape[0] != self.input_size[0] or image.shape[1] != self.input_size[1]:
            # keep aspect ratio when resize
            ratio_size = (np.array(image.shape[:2]) * ratio).astype(np.int32)
            image = cv.resize(image, (ratio_size[1], ratio_size[0]))
            pad_h = self.input_size[0] - ratio_size[0]
            pad_w = self.input_size[1] - ratio_size[1]
            pad_bias[0] = left = pad_w // 2
            pad_bias[1] = top = pad_h // 2
            right = pad_w - left
            bottom = pad_h - top
            image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0 # norm
        pad_bias = (pad_bias / ratio).astype(np.int32)
        return image[np.newaxis, :, :, :], pad_bias # hwc -> nhwc

    def infer(self, image):
        h, w, _ = image.shape

        # Preprocess
        input_blob, pad_bias = self._preprocess(image)

        # Forward
        self.model.setInput(input_blob)
        output_blob = self.model.forward(self.model.getUnconnectedOutLayersNames())

        # Postprocess
        results = self._postprocess(output_blob, np.array([w, h]), pad_bias)

        return results

    def _postprocess(self, output_blob, original_shape, pad_bias):
        score = output_blob[1][0, :, 0]
        box_delta = output_blob[0][0, :, 0:4]
        landmark_delta = output_blob[0][0, :, 4:]
        scale = max(original_shape)

        # get scores
        score = score.astype(np.float64)
        score = 1 / (1 + np.exp(-score))

        # get boxes
        cxy_delta = box_delta[:, :2] / self.input_size
        wh_delta = box_delta[:, 2:] / self.input_size
        xy1 = (cxy_delta - wh_delta / 2 + self.anchors) * scale
        xy2 = (cxy_delta + wh_delta / 2 + self.anchors) * scale
        boxes = np.concatenate([xy1, xy2], axis=1)
        boxes -= [pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]]
        # NMS
        keep_idx = cv.dnn.NMSBoxes(boxes, score, self.score_threshold, self.nms_threshold, top_k=self.topK)
        if len(keep_idx) == 0:
            return np.empty(shape=(0, 19))
        selected_score = score[keep_idx]
        selected_box = boxes[keep_idx]

        # get landmarks
        selected_landmarks = landmark_delta[keep_idx].reshape(-1, 7, 2)
        selected_landmarks = selected_landmarks / self.input_size
        selected_anchors = self.anchors[keep_idx]
        for idx, landmark in enumerate(selected_landmarks):
            landmark += selected_anchors[idx]
        selected_landmarks *= scale
        selected_landmarks -= pad_bias

        # [
        #   [bbox_coords, landmarks_coords, score]
        #   ...
        #   [bbox_coords, landmarks_coords, score]
        # ]
        return np.c_[selected_box.reshape(-1, 4), selected_landmarks.reshape(-1, 14), selected_score.reshape(-1, 1)]

    def _load_anchors(self):
        return np.array([[0.02083333, 0.02083333],
                      [0.02083333, 0.02083333],
                      [0.0625, 0.02083333],
                      [0.0625, 0.02083333],
                      [0.10416666, 0.02083333],
                      [0.10416666, 0.02083333],
                      [0.14583333, 0.02083333],
                      [0.14583333, 0.02083333],
                      [0.1875, 0.02083333],
                      [0.1875, 0.02083333],
                      [0.22916667, 0.02083333],
                      [0.22916667, 0.02083333],
                      [0.27083334, 0.02083333],
                      [0.27083334, 0.02083333],
                      [0.3125, 0.02083333],
                      [0.3125, 0.02083333],
                      [0.35416666, 0.02083333],
                      [0.35416666, 0.02083333],
                      [0.39583334, 0.02083333],
                      [0.39583334, 0.02083333],
                      [0.4375, 0.02083333],
                      [0.4375, 0.02083333],
                      [0.47916666, 0.02083333],
                      [0.47916666, 0.02083333],
                      [0.5208333, 0.02083333],
                      [0.5208333, 0.02083333],
                      [0.5625, 0.02083333],
                      [0.5625, 0.02083333],
                      [0.6041667, 0.02083333],
                      [0.6041667, 0.02083333],
                      [0.6458333, 0.02083333],
                      [0.6458333, 0.02083333],
                      [0.6875, 0.02083333],
                      [0.6875, 0.02083333],
                      [0.7291667, 0.02083333],
                      [0.7291667, 0.02083333],
                      [0.7708333, 0.02083333],
                      [0.7708333, 0.02083333],
                      [0.8125, 0.02083333],
                      [0.8125, 0.02083333],
                      [0.8541667, 0.02083333],
                      [0.8541667, 0.02083333],
                      [0.8958333, 0.02083333],
                      [0.8958333, 0.02083333],
                      [0.9375, 0.02083333],
                      [0.9375, 0.02083333],
                      [0.9791667, 0.02083333],
                      [0.9791667, 0.02083333],
                      [0.02083333, 0.0625],
                      [0.02083333, 0.0625],
                      [0.0625, 0.0625],
                      [0.0625, 0.0625],
                      [0.10416666, 0.0625],
                      [0.10416666, 0.0625],
                      [0.14583333, 0.0625],
                      [0.14583333, 0.0625],
                      [0.1875, 0.0625],
                      [0.1875, 0.0625],
                      [0.22916667, 0.0625],
                      [0.22916667, 0.0625],
                      [0.27083334, 0.0625],
                      [0.27083334, 0.0625],
                      [0.3125, 0.0625],
                      [0.3125, 0.0625],
                      [0.35416666, 0.0625],
                      [0.35416666, 0.0625],
                      [0.39583334, 0.0625],
                      [0.39583334, 0.0625],
                      [0.4375, 0.0625],
                      [0.4375, 0.0625],
                      [0.47916666, 0.0625],
                      [0.47916666, 0.0625],
                      [0.5208333, 0.0625],
                      [0.5208333, 0.0625],
                      [0.5625, 0.0625],
                      [0.5625, 0.0625],
                      [0.6041667, 0.0625],
                      [0.6041667, 0.0625],
                      [0.6458333, 0.0625],
                      [0.6458333, 0.0625],
                      [0.6875, 0.0625],
                      [0.6875, 0.0625],
                      [0.7291667, 0.0625],
                      [0.7291667, 0.0625],
                      [0.7708333, 0.0625],
                      [0.7708333, 0.0625],
                      [0.8125, 0.0625],
                      [0.8125, 0.0625],
                      [0.8541667, 0.0625],
                      [0.8541667, 0.0625],
                      [0.8958333, 0.0625],
                      [0.8958333, 0.0625],
                      [0.9375, 0.0625],
                      [0.9375, 0.0625],
                      [0.9791667, 0.0625],
                      [0.9791667, 0.0625],
                      [0.02083333, 0.10416666],
                      [0.02083333, 0.10416666],
                      [0.0625, 0.10416666],
                      [0.0625, 0.10416666],
                      [0.10416666, 0.10416666],
                      [0.10416666, 0.10416666],
                      [0.14583333, 0.10416666],
                      [0.14583333, 0.10416666],
                      [0.1875, 0.10416666],
                      [0.1875, 0.10416666],
                      [0.22916667, 0.10416666],
                      [0.22916667, 0.10416666],
                      [0.27083334, 0.10416666],
                      [0.27083334, 0.10416666],
                      [0.3125, 0.10416666],
                      [0.3125, 0.10416666],
                      [0.35416666, 0.10416666],
                      [0.35416666, 0.10416666],
                      [0.39583334, 0.10416666],
                      [0.39583334, 0.10416666],
                      [0.4375, 0.10416666],
                      [0.4375, 0.10416666],
                      [0.47916666, 0.10416666],
                      [0.47916666, 0.10416666],
                      [0.5208333, 0.10416666],
                      [0.5208333, 0.10416666],
                      [0.5625, 0.10416666],
                      [0.5625, 0.10416666],
                      [0.6041667, 0.10416666],
                      [0.6041667, 0.10416666],
                      [0.6458333, 0.10416666],
                      [0.6458333, 0.10416666],
                      [0.6875, 0.10416666],
                      [0.6875, 0.10416666],
                      [0.7291667, 0.10416666],
                      [0.7291667, 0.10416666],
                      [0.7708333, 0.10416666],
                      [0.7708333, 0.10416666],
                      [0.8125, 0.10416666],
                      [0.8125, 0.10416666],
                      [0.8541667, 0.10416666],
                      [0.8541667, 0.10416666],
                      [0.8958333, 0.10416666],
                      [0.8958333, 0.10416666],
                      [0.9375, 0.10416666],
                      [0.9375, 0.10416666],
                      [0.9791667, 0.10416666],
                      [0.9791667, 0.10416666],
                      [0.02083333, 0.14583333],
                      [0.02083333, 0.14583333],
                      [0.0625, 0.14583333],
                      [0.0625, 0.14583333],
                      [0.10416666, 0.14583333],
                      [0.10416666, 0.14583333],
                      [0.14583333, 0.14583333],
                      [0.14583333, 0.14583333],
                      [0.1875, 0.14583333],
                      [0.1875, 0.14583333],
                      [0.22916667, 0.14583333],
                      [0.22916667, 0.14583333],
                      [0.27083334, 0.14583333],
                      [0.27083334, 0.14583333],
                      [0.3125, 0.14583333],
                      [0.3125, 0.14583333],
                      [0.35416666, 0.14583333],
                      [0.35416666, 0.14583333],
                      [0.39583334, 0.14583333],
                      [0.39583334, 0.14583333],
                      [0.4375, 0.14583333],
                      [0.4375, 0.14583333],
                      [0.47916666, 0.14583333],
                      [0.47916666, 0.14583333],
                      [0.5208333, 0.14583333],
                      [0.5208333, 0.14583333],
                      [0.5625, 0.14583333],
                      [0.5625, 0.14583333],
                      [0.6041667, 0.14583333],
                      [0.6041667, 0.14583333],
                      [0.6458333, 0.14583333],
                      [0.6458333, 0.14583333],
                      [0.6875, 0.14583333],
                      [0.6875, 0.14583333],
                      [0.7291667, 0.14583333],
                      [0.7291667, 0.14583333],
                      [0.7708333, 0.14583333],
                      [0.7708333, 0.14583333],
                      [0.8125, 0.14583333],
                      [0.8125, 0.14583333],
                      [0.8541667, 0.14583333],
                      [0.8541667, 0.14583333],
                      [0.8958333, 0.14583333],
                      [0.8958333, 0.14583333],
                      [0.9375, 0.14583333],
                      [0.9375, 0.14583333],
                      [0.9791667, 0.14583333],
                      [0.9791667, 0.14583333],
                      [0.02083333, 0.1875],
                      [0.02083333, 0.1875],
                      [0.0625, 0.1875],
                      [0.0625, 0.1875],
                      [0.10416666, 0.1875],
                      [0.10416666, 0.1875],
                      [0.14583333, 0.1875],
                      [0.14583333, 0.1875],
                      [0.1875, 0.1875],
                      [0.1875, 0.1875],
                      [0.22916667, 0.1875],
                      [0.22916667, 0.1875],
                      [0.27083334, 0.1875],
                      [0.27083334, 0.1875],
                      [0.3125, 0.1875],
                      [0.3125, 0.1875],
                      [0.35416666, 0.1875],
                      [0.35416666, 0.1875],
                      [0.39583334, 0.1875],
                      [0.39583334, 0.1875],
                      [0.4375, 0.1875],
                      [0.4375, 0.1875],
                      [0.47916666, 0.1875],
                      [0.47916666, 0.1875],
                      [0.5208333, 0.1875],
                      [0.5208333, 0.1875],
                      [0.5625, 0.1875],
                      [0.5625, 0.1875],
                      [0.6041667, 0.1875],
                      [0.6041667, 0.1875],
                      [0.6458333, 0.1875],
                      [0.6458333, 0.1875],
                      [0.6875, 0.1875],
                      [0.6875, 0.1875],
                      [0.7291667, 0.1875],
                      [0.7291667, 0.1875],
                      [0.7708333, 0.1875],
                      [0.7708333, 0.1875],
                      [0.8125, 0.1875],
                      [0.8125, 0.1875],
                      [0.8541667, 0.1875],
                      [0.8541667, 0.1875],
                      [0.8958333, 0.1875],
                      [0.8958333, 0.1875],
                      [0.9375, 0.1875],
                      [0.9375, 0.1875],
                      [0.9791667, 0.1875],
                      [0.9791667, 0.1875],
                      [0.02083333, 0.22916667],
                      [0.02083333, 0.22916667],
                      [0.0625, 0.22916667],
                      [0.0625, 0.22916667],
                      [0.10416666, 0.22916667],
                      [0.10416666, 0.22916667],
                      [0.14583333, 0.22916667],
                      [0.14583333, 0.22916667],
                      [0.1875, 0.22916667],
                      [0.1875, 0.22916667],
                      [0.22916667, 0.22916667],
                      [0.22916667, 0.22916667],
                      [0.27083334, 0.22916667],
                      [0.27083334, 0.22916667],
                      [0.3125, 0.22916667],
                      [0.3125, 0.22916667],
                      [0.35416666, 0.22916667],
                      [0.35416666, 0.22916667],
                      [0.39583334, 0.22916667],
                      [0.39583334, 0.22916667],
                      [0.4375, 0.22916667],
                      [0.4375, 0.22916667],
                      [0.47916666, 0.22916667],
                      [0.47916666, 0.22916667],
                      [0.5208333, 0.22916667],
                      [0.5208333, 0.22916667],
                      [0.5625, 0.22916667],
                      [0.5625, 0.22916667],
                      [0.6041667, 0.22916667],
                      [0.6041667, 0.22916667],
                      [0.6458333, 0.22916667],
                      [0.6458333, 0.22916667],
                      [0.6875, 0.22916667],
                      [0.6875, 0.22916667],
                      [0.7291667, 0.22916667],
                      [0.7291667, 0.22916667],
                      [0.7708333, 0.22916667],
                      [0.7708333, 0.22916667],
                      [0.8125, 0.22916667],
                      [0.8125, 0.22916667],
                      [0.8541667, 0.22916667],
                      [0.8541667, 0.22916667],
                      [0.8958333, 0.22916667],
                      [0.8958333, 0.22916667],
                      [0.9375, 0.22916667],
                      [0.9375, 0.22916667],
                      [0.9791667, 0.22916667],
                      [0.9791667, 0.22916667],
                      [0.02083333, 0.27083334],
                      [0.02083333, 0.27083334],
                      [0.0625, 0.27083334],
                      [0.0625, 0.27083334],
                      [0.10416666, 0.27083334],
                      [0.10416666, 0.27083334],
                      [0.14583333, 0.27083334],
                      [0.14583333, 0.27083334],
                      [0.1875, 0.27083334],
                      [0.1875, 0.27083334],
                      [0.22916667, 0.27083334],
                      [0.22916667, 0.27083334],
                      [0.27083334, 0.27083334],
                      [0.27083334, 0.27083334],
                      [0.3125, 0.27083334],
                      [0.3125, 0.27083334],
                      [0.35416666, 0.27083334],
                      [0.35416666, 0.27083334],
                      [0.39583334, 0.27083334],
                      [0.39583334, 0.27083334],
                      [0.4375, 0.27083334],
                      [0.4375, 0.27083334],
                      [0.47916666, 0.27083334],
                      [0.47916666, 0.27083334],
                      [0.5208333, 0.27083334],
                      [0.5208333, 0.27083334],
                      [0.5625, 0.27083334],
                      [0.5625, 0.27083334],
                      [0.6041667, 0.27083334],
                      [0.6041667, 0.27083334],
                      [0.6458333, 0.27083334],
                      [0.6458333, 0.27083334],
                      [0.6875, 0.27083334],
                      [0.6875, 0.27083334],
                      [0.7291667, 0.27083334],
                      [0.7291667, 0.27083334],
                      [0.7708333, 0.27083334],
                      [0.7708333, 0.27083334],
                      [0.8125, 0.27083334],
                      [0.8125, 0.27083334],
                      [0.8541667, 0.27083334],
                      [0.8541667, 0.27083334],
                      [0.8958333, 0.27083334],
                      [0.8958333, 0.27083334],
                      [0.9375, 0.27083334],
                      [0.9375, 0.27083334],
                      [0.9791667, 0.27083334],
                      [0.9791667, 0.27083334],
                      [0.02083333, 0.3125],
                      [0.02083333, 0.3125],
                      [0.0625, 0.3125],
                      [0.0625, 0.3125],
                      [0.10416666, 0.3125],
                      [0.10416666, 0.3125],
                      [0.14583333, 0.3125],
                      [0.14583333, 0.3125],
                      [0.1875, 0.3125],
                      [0.1875, 0.3125],
                      [0.22916667, 0.3125],
                      [0.22916667, 0.3125],
                      [0.27083334, 0.3125],
                      [0.27083334, 0.3125],
                      [0.3125, 0.3125],
                      [0.3125, 0.3125],
                      [0.35416666, 0.3125],
                      [0.35416666, 0.3125],
                      [0.39583334, 0.3125],
                      [0.39583334, 0.3125],
                      [0.4375, 0.3125],
                      [0.4375, 0.3125],
                      [0.47916666, 0.3125],
                      [0.47916666, 0.3125],
                      [0.5208333, 0.3125],
                      [0.5208333, 0.3125],
                      [0.5625, 0.3125],
                      [0.5625, 0.3125],
                      [0.6041667, 0.3125],
                      [0.6041667, 0.3125],
                      [0.6458333, 0.3125],
                      [0.6458333, 0.3125],
                      [0.6875, 0.3125],
                      [0.6875, 0.3125],
                      [0.7291667, 0.3125],
                      [0.7291667, 0.3125],
                      [0.7708333, 0.3125],
                      [0.7708333, 0.3125],
                      [0.8125, 0.3125],
                      [0.8125, 0.3125],
                      [0.8541667, 0.3125],
                      [0.8541667, 0.3125],
                      [0.8958333, 0.3125],
                      [0.8958333, 0.3125],
                      [0.9375, 0.3125],
                      [0.9375, 0.3125],
                      [0.9791667, 0.3125],
                      [0.9791667, 0.3125],
                      [0.02083333, 0.35416666],
                      [0.02083333, 0.35416666],
                      [0.0625, 0.35416666],
                      [0.0625, 0.35416666],
                      [0.10416666, 0.35416666],
                      [0.10416666, 0.35416666],
                      [0.14583333, 0.35416666],
                      [0.14583333, 0.35416666],
                      [0.1875, 0.35416666],
                      [0.1875, 0.35416666],
                      [0.22916667, 0.35416666],
                      [0.22916667, 0.35416666],
                      [0.27083334, 0.35416666],
                      [0.27083334, 0.35416666],
                      [0.3125, 0.35416666],
                      [0.3125, 0.35416666],
                      [0.35416666, 0.35416666],
                      [0.35416666, 0.35416666],
                      [0.39583334, 0.35416666],
                      [0.39583334, 0.35416666],
                      [0.4375, 0.35416666],
                      [0.4375, 0.35416666],
                      [0.47916666, 0.35416666],
                      [0.47916666, 0.35416666],
                      [0.5208333, 0.35416666],
                      [0.5208333, 0.35416666],
                      [0.5625, 0.35416666],
                      [0.5625, 0.35416666],
                      [0.6041667, 0.35416666],
                      [0.6041667, 0.35416666],
                      [0.6458333, 0.35416666],
                      [0.6458333, 0.35416666],
                      [0.6875, 0.35416666],
                      [0.6875, 0.35416666],
                      [0.7291667, 0.35416666],
                      [0.7291667, 0.35416666],
                      [0.7708333, 0.35416666],
                      [0.7708333, 0.35416666],
                      [0.8125, 0.35416666],
                      [0.8125, 0.35416666],
                      [0.8541667, 0.35416666],
                      [0.8541667, 0.35416666],
                      [0.8958333, 0.35416666],
                      [0.8958333, 0.35416666],
                      [0.9375, 0.35416666],
                      [0.9375, 0.35416666],
                      [0.9791667, 0.35416666],
                      [0.9791667, 0.35416666],
                      [0.02083333, 0.39583334],
                      [0.02083333, 0.39583334],
                      [0.0625, 0.39583334],
                      [0.0625, 0.39583334],
                      [0.10416666, 0.39583334],
                      [0.10416666, 0.39583334],
                      [0.14583333, 0.39583334],
                      [0.14583333, 0.39583334],
                      [0.1875, 0.39583334],
                      [0.1875, 0.39583334],
                      [0.22916667, 0.39583334],
                      [0.22916667, 0.39583334],
                      [0.27083334, 0.39583334],
                      [0.27083334, 0.39583334],
                      [0.3125, 0.39583334],
                      [0.3125, 0.39583334],
                      [0.35416666, 0.39583334],
                      [0.35416666, 0.39583334],
                      [0.39583334, 0.39583334],
                      [0.39583334, 0.39583334],
                      [0.4375, 0.39583334],
                      [0.4375, 0.39583334],
                      [0.47916666, 0.39583334],
                      [0.47916666, 0.39583334],
                      [0.5208333, 0.39583334],
                      [0.5208333, 0.39583334],
                      [0.5625, 0.39583334],
                      [0.5625, 0.39583334],
                      [0.6041667, 0.39583334],
                      [0.6041667, 0.39583334],
                      [0.6458333, 0.39583334],
                      [0.6458333, 0.39583334],
                      [0.6875, 0.39583334],
                      [0.6875, 0.39583334],
                      [0.7291667, 0.39583334],
                      [0.7291667, 0.39583334],
                      [0.7708333, 0.39583334],
                      [0.7708333, 0.39583334],
                      [0.8125, 0.39583334],
                      [0.8125, 0.39583334],
                      [0.8541667, 0.39583334],
                      [0.8541667, 0.39583334],
                      [0.8958333, 0.39583334],
                      [0.8958333, 0.39583334],
                      [0.9375, 0.39583334],
                      [0.9375, 0.39583334],
                      [0.9791667, 0.39583334],
                      [0.9791667, 0.39583334],
                      [0.02083333, 0.4375],
                      [0.02083333, 0.4375],
                      [0.0625, 0.4375],
                      [0.0625, 0.4375],
                      [0.10416666, 0.4375],
                      [0.10416666, 0.4375],
                      [0.14583333, 0.4375],
                      [0.14583333, 0.4375],
                      [0.1875, 0.4375],
                      [0.1875, 0.4375],
                      [0.22916667, 0.4375],
                      [0.22916667, 0.4375],
                      [0.27083334, 0.4375],
                      [0.27083334, 0.4375],
                      [0.3125, 0.4375],
                      [0.3125, 0.4375],
                      [0.35416666, 0.4375],
                      [0.35416666, 0.4375],
                      [0.39583334, 0.4375],
                      [0.39583334, 0.4375],
                      [0.4375, 0.4375],
                      [0.4375, 0.4375],
                      [0.47916666, 0.4375],
                      [0.47916666, 0.4375],
                      [0.5208333, 0.4375],
                      [0.5208333, 0.4375],
                      [0.5625, 0.4375],
                      [0.5625, 0.4375],
                      [0.6041667, 0.4375],
                      [0.6041667, 0.4375],
                      [0.6458333, 0.4375],
                      [0.6458333, 0.4375],
                      [0.6875, 0.4375],
                      [0.6875, 0.4375],
                      [0.7291667, 0.4375],
                      [0.7291667, 0.4375],
                      [0.7708333, 0.4375],
                      [0.7708333, 0.4375],
                      [0.8125, 0.4375],
                      [0.8125, 0.4375],
                      [0.8541667, 0.4375],
                      [0.8541667, 0.4375],
                      [0.8958333, 0.4375],
                      [0.8958333, 0.4375],
                      [0.9375, 0.4375],
                      [0.9375, 0.4375],
                      [0.9791667, 0.4375],
                      [0.9791667, 0.4375],
                      [0.02083333, 0.47916666],
                      [0.02083333, 0.47916666],
                      [0.0625, 0.47916666],
                      [0.0625, 0.47916666],
                      [0.10416666, 0.47916666],
                      [0.10416666, 0.47916666],
                      [0.14583333, 0.47916666],
                      [0.14583333, 0.47916666],
                      [0.1875, 0.47916666],
                      [0.1875, 0.47916666],
                      [0.22916667, 0.47916666],
                      [0.22916667, 0.47916666],
                      [0.27083334, 0.47916666],
                      [0.27083334, 0.47916666],
                      [0.3125, 0.47916666],
                      [0.3125, 0.47916666],
                      [0.35416666, 0.47916666],
                      [0.35416666, 0.47916666],
                      [0.39583334, 0.47916666],
                      [0.39583334, 0.47916666],
                      [0.4375, 0.47916666],
                      [0.4375, 0.47916666],
                      [0.47916666, 0.47916666],
                      [0.47916666, 0.47916666],
                      [0.5208333, 0.47916666],
                      [0.5208333, 0.47916666],
                      [0.5625, 0.47916666],
                      [0.5625, 0.47916666],
                      [0.6041667, 0.47916666],
                      [0.6041667, 0.47916666],
                      [0.6458333, 0.47916666],
                      [0.6458333, 0.47916666],
                      [0.6875, 0.47916666],
                      [0.6875, 0.47916666],
                      [0.7291667, 0.47916666],
                      [0.7291667, 0.47916666],
                      [0.7708333, 0.47916666],
                      [0.7708333, 0.47916666],
                      [0.8125, 0.47916666],
                      [0.8125, 0.47916666],
                      [0.8541667, 0.47916666],
                      [0.8541667, 0.47916666],
                      [0.8958333, 0.47916666],
                      [0.8958333, 0.47916666],
                      [0.9375, 0.47916666],
                      [0.9375, 0.47916666],
                      [0.9791667, 0.47916666],
                      [0.9791667, 0.47916666],
                      [0.02083333, 0.5208333],
                      [0.02083333, 0.5208333],
                      [0.0625, 0.5208333],
                      [0.0625, 0.5208333],
                      [0.10416666, 0.5208333],
                      [0.10416666, 0.5208333],
                      [0.14583333, 0.5208333],
                      [0.14583333, 0.5208333],
                      [0.1875, 0.5208333],
                      [0.1875, 0.5208333],
                      [0.22916667, 0.5208333],
                      [0.22916667, 0.5208333],
                      [0.27083334, 0.5208333],
                      [0.27083334, 0.5208333],
                      [0.3125, 0.5208333],
                      [0.3125, 0.5208333],
                      [0.35416666, 0.5208333],
                      [0.35416666, 0.5208333],
                      [0.39583334, 0.5208333],
                      [0.39583334, 0.5208333],
                      [0.4375, 0.5208333],
                      [0.4375, 0.5208333],
                      [0.47916666, 0.5208333],
                      [0.47916666, 0.5208333],
                      [0.5208333, 0.5208333],
                      [0.5208333, 0.5208333],
                      [0.5625, 0.5208333],
                      [0.5625, 0.5208333],
                      [0.6041667, 0.5208333],
                      [0.6041667, 0.5208333],
                      [0.6458333, 0.5208333],
                      [0.6458333, 0.5208333],
                      [0.6875, 0.5208333],
                      [0.6875, 0.5208333],
                      [0.7291667, 0.5208333],
                      [0.7291667, 0.5208333],
                      [0.7708333, 0.5208333],
                      [0.7708333, 0.5208333],
                      [0.8125, 0.5208333],
                      [0.8125, 0.5208333],
                      [0.8541667, 0.5208333],
                      [0.8541667, 0.5208333],
                      [0.8958333, 0.5208333],
                      [0.8958333, 0.5208333],
                      [0.9375, 0.5208333],
                      [0.9375, 0.5208333],
                      [0.9791667, 0.5208333],
                      [0.9791667, 0.5208333],
                      [0.02083333, 0.5625],
                      [0.02083333, 0.5625],
                      [0.0625, 0.5625],
                      [0.0625, 0.5625],
                      [0.10416666, 0.5625],
                      [0.10416666, 0.5625],
                      [0.14583333, 0.5625],
                      [0.14583333, 0.5625],
                      [0.1875, 0.5625],
                      [0.1875, 0.5625],
                      [0.22916667, 0.5625],
                      [0.22916667, 0.5625],
                      [0.27083334, 0.5625],
                      [0.27083334, 0.5625],
                      [0.3125, 0.5625],
                      [0.3125, 0.5625],
                      [0.35416666, 0.5625],
                      [0.35416666, 0.5625],
                      [0.39583334, 0.5625],
                      [0.39583334, 0.5625],
                      [0.4375, 0.5625],
                      [0.4375, 0.5625],
                      [0.47916666, 0.5625],
                      [0.47916666, 0.5625],
                      [0.5208333, 0.5625],
                      [0.5208333, 0.5625],
                      [0.5625, 0.5625],
                      [0.5625, 0.5625],
                      [0.6041667, 0.5625],
                      [0.6041667, 0.5625],
                      [0.6458333, 0.5625],
                      [0.6458333, 0.5625],
                      [0.6875, 0.5625],
                      [0.6875, 0.5625],
                      [0.7291667, 0.5625],
                      [0.7291667, 0.5625],
                      [0.7708333, 0.5625],
                      [0.7708333, 0.5625],
                      [0.8125, 0.5625],
                      [0.8125, 0.5625],
                      [0.8541667, 0.5625],
                      [0.8541667, 0.5625],
                      [0.8958333, 0.5625],
                      [0.8958333, 0.5625],
                      [0.9375, 0.5625],
                      [0.9375, 0.5625],
                      [0.9791667, 0.5625],
                      [0.9791667, 0.5625],
                      [0.02083333, 0.6041667],
                      [0.02083333, 0.6041667],
                      [0.0625, 0.6041667],
                      [0.0625, 0.6041667],
                      [0.10416666, 0.6041667],
                      [0.10416666, 0.6041667],
                      [0.14583333, 0.6041667],
                      [0.14583333, 0.6041667],
                      [0.1875, 0.6041667],
                      [0.1875, 0.6041667],
                      [0.22916667, 0.6041667],
                      [0.22916667, 0.6041667],
                      [0.27083334, 0.6041667],
                      [0.27083334, 0.6041667],
                      [0.3125, 0.6041667],
                      [0.3125, 0.6041667],
                      [0.35416666, 0.6041667],
                      [0.35416666, 0.6041667],
                      [0.39583334, 0.6041667],
                      [0.39583334, 0.6041667],
                      [0.4375, 0.6041667],
                      [0.4375, 0.6041667],
                      [0.47916666, 0.6041667],
                      [0.47916666, 0.6041667],
                      [0.5208333, 0.6041667],
                      [0.5208333, 0.6041667],
                      [0.5625, 0.6041667],
                      [0.5625, 0.6041667],
                      [0.6041667, 0.6041667],
                      [0.6041667, 0.6041667],
                      [0.6458333, 0.6041667],
                      [0.6458333, 0.6041667],
                      [0.6875, 0.6041667],
                      [0.6875, 0.6041667],
                      [0.7291667, 0.6041667],
                      [0.7291667, 0.6041667],
                      [0.7708333, 0.6041667],
                      [0.7708333, 0.6041667],
                      [0.8125, 0.6041667],
                      [0.8125, 0.6041667],
                      [0.8541667, 0.6041667],
                      [0.8541667, 0.6041667],
                      [0.8958333, 0.6041667],
                      [0.8958333, 0.6041667],
                      [0.9375, 0.6041667],
                      [0.9375, 0.6041667],
                      [0.9791667, 0.6041667],
                      [0.9791667, 0.6041667],
                      [0.02083333, 0.6458333],
                      [0.02083333, 0.6458333],
                      [0.0625, 0.6458333],
                      [0.0625, 0.6458333],
                      [0.10416666, 0.6458333],
                      [0.10416666, 0.6458333],
                      [0.14583333, 0.6458333],
                      [0.14583333, 0.6458333],
                      [0.1875, 0.6458333],
                      [0.1875, 0.6458333],
                      [0.22916667, 0.6458333],
                      [0.22916667, 0.6458333],
                      [0.27083334, 0.6458333],
                      [0.27083334, 0.6458333],
                      [0.3125, 0.6458333],
                      [0.3125, 0.6458333],
                      [0.35416666, 0.6458333],
                      [0.35416666, 0.6458333],
                      [0.39583334, 0.6458333],
                      [0.39583334, 0.6458333],
                      [0.4375, 0.6458333],
                      [0.4375, 0.6458333],
                      [0.47916666, 0.6458333],
                      [0.47916666, 0.6458333],
                      [0.5208333, 0.6458333],
                      [0.5208333, 0.6458333],
                      [0.5625, 0.6458333],
                      [0.5625, 0.6458333],
                      [0.6041667, 0.6458333],
                      [0.6041667, 0.6458333],
                      [0.6458333, 0.6458333],
                      [0.6458333, 0.6458333],
                      [0.6875, 0.6458333],
                      [0.6875, 0.6458333],
                      [0.7291667, 0.6458333],
                      [0.7291667, 0.6458333],
                      [0.7708333, 0.6458333],
                      [0.7708333, 0.6458333],
                      [0.8125, 0.6458333],
                      [0.8125, 0.6458333],
                      [0.8541667, 0.6458333],
                      [0.8541667, 0.6458333],
                      [0.8958333, 0.6458333],
                      [0.8958333, 0.6458333],
                      [0.9375, 0.6458333],
                      [0.9375, 0.6458333],
                      [0.9791667, 0.6458333],
                      [0.9791667, 0.6458333],
                      [0.02083333, 0.6875],
                      [0.02083333, 0.6875],
                      [0.0625, 0.6875],
                      [0.0625, 0.6875],
                      [0.10416666, 0.6875],
                      [0.10416666, 0.6875],
                      [0.14583333, 0.6875],
                      [0.14583333, 0.6875],
                      [0.1875, 0.6875],
                      [0.1875, 0.6875],
                      [0.22916667, 0.6875],
                      [0.22916667, 0.6875],
                      [0.27083334, 0.6875],
                      [0.27083334, 0.6875],
                      [0.3125, 0.6875],
                      [0.3125, 0.6875],
                      [0.35416666, 0.6875],
                      [0.35416666, 0.6875],
                      [0.39583334, 0.6875],
                      [0.39583334, 0.6875],
                      [0.4375, 0.6875],
                      [0.4375, 0.6875],
                      [0.47916666, 0.6875],
                      [0.47916666, 0.6875],
                      [0.5208333, 0.6875],
                      [0.5208333, 0.6875],
                      [0.5625, 0.6875],
                      [0.5625, 0.6875],
                      [0.6041667, 0.6875],
                      [0.6041667, 0.6875],
                      [0.6458333, 0.6875],
                      [0.6458333, 0.6875],
                      [0.6875, 0.6875],
                      [0.6875, 0.6875],
                      [0.7291667, 0.6875],
                      [0.7291667, 0.6875],
                      [0.7708333, 0.6875],
                      [0.7708333, 0.6875],
                      [0.8125, 0.6875],
                      [0.8125, 0.6875],
                      [0.8541667, 0.6875],
                      [0.8541667, 0.6875],
                      [0.8958333, 0.6875],
                      [0.8958333, 0.6875],
                      [0.9375, 0.6875],
                      [0.9375, 0.6875],
                      [0.9791667, 0.6875],
                      [0.9791667, 0.6875],
                      [0.02083333, 0.7291667],
                      [0.02083333, 0.7291667],
                      [0.0625, 0.7291667],
                      [0.0625, 0.7291667],
                      [0.10416666, 0.7291667],
                      [0.10416666, 0.7291667],
                      [0.14583333, 0.7291667],
                      [0.14583333, 0.7291667],
                      [0.1875, 0.7291667],
                      [0.1875, 0.7291667],
                      [0.22916667, 0.7291667],
                      [0.22916667, 0.7291667],
                      [0.27083334, 0.7291667],
                      [0.27083334, 0.7291667],
                      [0.3125, 0.7291667],
                      [0.3125, 0.7291667],
                      [0.35416666, 0.7291667],
                      [0.35416666, 0.7291667],
                      [0.39583334, 0.7291667],
                      [0.39583334, 0.7291667],
                      [0.4375, 0.7291667],
                      [0.4375, 0.7291667],
                      [0.47916666, 0.7291667],
                      [0.47916666, 0.7291667],
                      [0.5208333, 0.7291667],
                      [0.5208333, 0.7291667],
                      [0.5625, 0.7291667],
                      [0.5625, 0.7291667],
                      [0.6041667, 0.7291667],
                      [0.6041667, 0.7291667],
                      [0.6458333, 0.7291667],
                      [0.6458333, 0.7291667],
                      [0.6875, 0.7291667],
                      [0.6875, 0.7291667],
                      [0.7291667, 0.7291667],
                      [0.7291667, 0.7291667],
                      [0.7708333, 0.7291667],
                      [0.7708333, 0.7291667],
                      [0.8125, 0.7291667],
                      [0.8125, 0.7291667],
                      [0.8541667, 0.7291667],
                      [0.8541667, 0.7291667],
                      [0.8958333, 0.7291667],
                      [0.8958333, 0.7291667],
                      [0.9375, 0.7291667],
                      [0.9375, 0.7291667],
                      [0.9791667, 0.7291667],
                      [0.9791667, 0.7291667],
                      [0.02083333, 0.7708333],
                      [0.02083333, 0.7708333],
                      [0.0625, 0.7708333],
                      [0.0625, 0.7708333],
                      [0.10416666, 0.7708333],
                      [0.10416666, 0.7708333],
                      [0.14583333, 0.7708333],
                      [0.14583333, 0.7708333],
                      [0.1875, 0.7708333],
                      [0.1875, 0.7708333],
                      [0.22916667, 0.7708333],
                      [0.22916667, 0.7708333],
                      [0.27083334, 0.7708333],
                      [0.27083334, 0.7708333],
                      [0.3125, 0.7708333],
                      [0.3125, 0.7708333],
                      [0.35416666, 0.7708333],
                      [0.35416666, 0.7708333],
                      [0.39583334, 0.7708333],
                      [0.39583334, 0.7708333],
                      [0.4375, 0.7708333],
                      [0.4375, 0.7708333],
                      [0.47916666, 0.7708333],
                      [0.47916666, 0.7708333],
                      [0.5208333, 0.7708333],
                      [0.5208333, 0.7708333],
                      [0.5625, 0.7708333],
                      [0.5625, 0.7708333],
                      [0.6041667, 0.7708333],
                      [0.6041667, 0.7708333],
                      [0.6458333, 0.7708333],
                      [0.6458333, 0.7708333],
                      [0.6875, 0.7708333],
                      [0.6875, 0.7708333],
                      [0.7291667, 0.7708333],
                      [0.7291667, 0.7708333],
                      [0.7708333, 0.7708333],
                      [0.7708333, 0.7708333],
                      [0.8125, 0.7708333],
                      [0.8125, 0.7708333],
                      [0.8541667, 0.7708333],
                      [0.8541667, 0.7708333],
                      [0.8958333, 0.7708333],
                      [0.8958333, 0.7708333],
                      [0.9375, 0.7708333],
                      [0.9375, 0.7708333],
                      [0.9791667, 0.7708333],
                      [0.9791667, 0.7708333],
                      [0.02083333, 0.8125],
                      [0.02083333, 0.8125],
                      [0.0625, 0.8125],
                      [0.0625, 0.8125],
                      [0.10416666, 0.8125],
                      [0.10416666, 0.8125],
                      [0.14583333, 0.8125],
                      [0.14583333, 0.8125],
                      [0.1875, 0.8125],
                      [0.1875, 0.8125],
                      [0.22916667, 0.8125],
                      [0.22916667, 0.8125],
                      [0.27083334, 0.8125],
                      [0.27083334, 0.8125],
                      [0.3125, 0.8125],
                      [0.3125, 0.8125],
                      [0.35416666, 0.8125],
                      [0.35416666, 0.8125],
                      [0.39583334, 0.8125],
                      [0.39583334, 0.8125],
                      [0.4375, 0.8125],
                      [0.4375, 0.8125],
                      [0.47916666, 0.8125],
                      [0.47916666, 0.8125],
                      [0.5208333, 0.8125],
                      [0.5208333, 0.8125],
                      [0.5625, 0.8125],
                      [0.5625, 0.8125],
                      [0.6041667, 0.8125],
                      [0.6041667, 0.8125],
                      [0.6458333, 0.8125],
                      [0.6458333, 0.8125],
                      [0.6875, 0.8125],
                      [0.6875, 0.8125],
                      [0.7291667, 0.8125],
                      [0.7291667, 0.8125],
                      [0.7708333, 0.8125],
                      [0.7708333, 0.8125],
                      [0.8125, 0.8125],
                      [0.8125, 0.8125],
                      [0.8541667, 0.8125],
                      [0.8541667, 0.8125],
                      [0.8958333, 0.8125],
                      [0.8958333, 0.8125],
                      [0.9375, 0.8125],
                      [0.9375, 0.8125],
                      [0.9791667, 0.8125],
                      [0.9791667, 0.8125],
                      [0.02083333, 0.8541667],
                      [0.02083333, 0.8541667],
                      [0.0625, 0.8541667],
                      [0.0625, 0.8541667],
                      [0.10416666, 0.8541667],
                      [0.10416666, 0.8541667],
                      [0.14583333, 0.8541667],
                      [0.14583333, 0.8541667],
                      [0.1875, 0.8541667],
                      [0.1875, 0.8541667],
                      [0.22916667, 0.8541667],
                      [0.22916667, 0.8541667],
                      [0.27083334, 0.8541667],
                      [0.27083334, 0.8541667],
                      [0.3125, 0.8541667],
                      [0.3125, 0.8541667],
                      [0.35416666, 0.8541667],
                      [0.35416666, 0.8541667],
                      [0.39583334, 0.8541667],
                      [0.39583334, 0.8541667],
                      [0.4375, 0.8541667],
                      [0.4375, 0.8541667],
                      [0.47916666, 0.8541667],
                      [0.47916666, 0.8541667],
                      [0.5208333, 0.8541667],
                      [0.5208333, 0.8541667],
                      [0.5625, 0.8541667],
                      [0.5625, 0.8541667],
                      [0.6041667, 0.8541667],
                      [0.6041667, 0.8541667],
                      [0.6458333, 0.8541667],
                      [0.6458333, 0.8541667],
                      [0.6875, 0.8541667],
                      [0.6875, 0.8541667],
                      [0.7291667, 0.8541667],
                      [0.7291667, 0.8541667],
                      [0.7708333, 0.8541667],
                      [0.7708333, 0.8541667],
                      [0.8125, 0.8541667],
                      [0.8125, 0.8541667],
                      [0.8541667, 0.8541667],
                      [0.8541667, 0.8541667],
                      [0.8958333, 0.8541667],
                      [0.8958333, 0.8541667],
                      [0.9375, 0.8541667],
                      [0.9375, 0.8541667],
                      [0.9791667, 0.8541667],
                      [0.9791667, 0.8541667],
                      [0.02083333, 0.8958333],
                      [0.02083333, 0.8958333],
                      [0.0625, 0.8958333],
                      [0.0625, 0.8958333],
                      [0.10416666, 0.8958333],
                      [0.10416666, 0.8958333],
                      [0.14583333, 0.8958333],
                      [0.14583333, 0.8958333],
                      [0.1875, 0.8958333],
                      [0.1875, 0.8958333],
                      [0.22916667, 0.8958333],
                      [0.22916667, 0.8958333],
                      [0.27083334, 0.8958333],
                      [0.27083334, 0.8958333],
                      [0.3125, 0.8958333],
                      [0.3125, 0.8958333],
                      [0.35416666, 0.8958333],
                      [0.35416666, 0.8958333],
                      [0.39583334, 0.8958333],
                      [0.39583334, 0.8958333],
                      [0.4375, 0.8958333],
                      [0.4375, 0.8958333],
                      [0.47916666, 0.8958333],
                      [0.47916666, 0.8958333],
                      [0.5208333, 0.8958333],
                      [0.5208333, 0.8958333],
                      [0.5625, 0.8958333],
                      [0.5625, 0.8958333],
                      [0.6041667, 0.8958333],
                      [0.6041667, 0.8958333],
                      [0.6458333, 0.8958333],
                      [0.6458333, 0.8958333],
                      [0.6875, 0.8958333],
                      [0.6875, 0.8958333],
                      [0.7291667, 0.8958333],
                      [0.7291667, 0.8958333],
                      [0.7708333, 0.8958333],
                      [0.7708333, 0.8958333],
                      [0.8125, 0.8958333],
                      [0.8125, 0.8958333],
                      [0.8541667, 0.8958333],
                      [0.8541667, 0.8958333],
                      [0.8958333, 0.8958333],
                      [0.8958333, 0.8958333],
                      [0.9375, 0.8958333],
                      [0.9375, 0.8958333],
                      [0.9791667, 0.8958333],
                      [0.9791667, 0.8958333],
                      [0.02083333, 0.9375],
                      [0.02083333, 0.9375],
                      [0.0625, 0.9375],
                      [0.0625, 0.9375],
                      [0.10416666, 0.9375],
                      [0.10416666, 0.9375],
                      [0.14583333, 0.9375],
                      [0.14583333, 0.9375],
                      [0.1875, 0.9375],
                      [0.1875, 0.9375],
                      [0.22916667, 0.9375],
                      [0.22916667, 0.9375],
                      [0.27083334, 0.9375],
                      [0.27083334, 0.9375],
                      [0.3125, 0.9375],
                      [0.3125, 0.9375],
                      [0.35416666, 0.9375],
                      [0.35416666, 0.9375],
                      [0.39583334, 0.9375],
                      [0.39583334, 0.9375],
                      [0.4375, 0.9375],
                      [0.4375, 0.9375],
                      [0.47916666, 0.9375],
                      [0.47916666, 0.9375],
                      [0.5208333, 0.9375],
                      [0.5208333, 0.9375],
                      [0.5625, 0.9375],
                      [0.5625, 0.9375],
                      [0.6041667, 0.9375],
                      [0.6041667, 0.9375],
                      [0.6458333, 0.9375],
                      [0.6458333, 0.9375],
                      [0.6875, 0.9375],
                      [0.6875, 0.9375],
                      [0.7291667, 0.9375],
                      [0.7291667, 0.9375],
                      [0.7708333, 0.9375],
                      [0.7708333, 0.9375],
                      [0.8125, 0.9375],
                      [0.8125, 0.9375],
                      [0.8541667, 0.9375],
                      [0.8541667, 0.9375],
                      [0.8958333, 0.9375],
                      [0.8958333, 0.9375],
                      [0.9375, 0.9375],
                      [0.9375, 0.9375],
                      [0.9791667, 0.9375],
                      [0.9791667, 0.9375],
                      [0.02083333, 0.9791667],
                      [0.02083333, 0.9791667],
                      [0.0625, 0.9791667],
                      [0.0625, 0.9791667],
                      [0.10416666, 0.9791667],
                      [0.10416666, 0.9791667],
                      [0.14583333, 0.9791667],
                      [0.14583333, 0.9791667],
                      [0.1875, 0.9791667],
                      [0.1875, 0.9791667],
                      [0.22916667, 0.9791667],
                      [0.22916667, 0.9791667],
                      [0.27083334, 0.9791667],
                      [0.27083334, 0.9791667],
                      [0.3125, 0.9791667],
                      [0.3125, 0.9791667],
                      [0.35416666, 0.9791667],
                      [0.35416666, 0.9791667],
                      [0.39583334, 0.9791667],
                      [0.39583334, 0.9791667],
                      [0.4375, 0.9791667],
                      [0.4375, 0.9791667],
                      [0.47916666, 0.9791667],
                      [0.47916666, 0.9791667],
                      [0.5208333, 0.9791667],
                      [0.5208333, 0.9791667],
                      [0.5625, 0.9791667],
                      [0.5625, 0.9791667],
                      [0.6041667, 0.9791667],
                      [0.6041667, 0.9791667],
                      [0.6458333, 0.9791667],
                      [0.6458333, 0.9791667],
                      [0.6875, 0.9791667],
                      [0.6875, 0.9791667],
                      [0.7291667, 0.9791667],
                      [0.7291667, 0.9791667],
                      [0.7708333, 0.9791667],
                      [0.7708333, 0.9791667],
                      [0.8125, 0.9791667],
                      [0.8125, 0.9791667],
                      [0.8541667, 0.9791667],
                      [0.8541667, 0.9791667],
                      [0.8958333, 0.9791667],
                      [0.8958333, 0.9791667],
                      [0.9375, 0.9791667],
                      [0.9375, 0.9791667],
                      [0.9791667, 0.9791667],
                      [0.9791667, 0.9791667],
                      [0.04166667, 0.04166667],
                      [0.04166667, 0.04166667],
                      [0.04166667, 0.04166667],
                      [0.04166667, 0.04166667],
                      [0.04166667, 0.04166667],
                      [0.04166667, 0.04166667],
                      [0.125, 0.04166667],
                      [0.125, 0.04166667],
                      [0.125, 0.04166667],
                      [0.125, 0.04166667],
                      [0.125, 0.04166667],
                      [0.125, 0.04166667],
                      [0.20833333, 0.04166667],
                      [0.20833333, 0.04166667],
                      [0.20833333, 0.04166667],
                      [0.20833333, 0.04166667],
                      [0.20833333, 0.04166667],
                      [0.20833333, 0.04166667],
                      [0.29166666, 0.04166667],
                      [0.29166666, 0.04166667],
                      [0.29166666, 0.04166667],
                      [0.29166666, 0.04166667],
                      [0.29166666, 0.04166667],
                      [0.29166666, 0.04166667],
                      [0.375, 0.04166667],
                      [0.375, 0.04166667],
                      [0.375, 0.04166667],
                      [0.375, 0.04166667],
                      [0.375, 0.04166667],
                      [0.375, 0.04166667],
                      [0.45833334, 0.04166667],
                      [0.45833334, 0.04166667],
                      [0.45833334, 0.04166667],
                      [0.45833334, 0.04166667],
                      [0.45833334, 0.04166667],
                      [0.45833334, 0.04166667],
                      [0.5416667, 0.04166667],
                      [0.5416667, 0.04166667],
                      [0.5416667, 0.04166667],
                      [0.5416667, 0.04166667],
                      [0.5416667, 0.04166667],
                      [0.5416667, 0.04166667],
                      [0.625, 0.04166667],
                      [0.625, 0.04166667],
                      [0.625, 0.04166667],
                      [0.625, 0.04166667],
                      [0.625, 0.04166667],
                      [0.625, 0.04166667],
                      [0.7083333, 0.04166667],
                      [0.7083333, 0.04166667],
                      [0.7083333, 0.04166667],
                      [0.7083333, 0.04166667],
                      [0.7083333, 0.04166667],
                      [0.7083333, 0.04166667],
                      [0.7916667, 0.04166667],
                      [0.7916667, 0.04166667],
                      [0.7916667, 0.04166667],
                      [0.7916667, 0.04166667],
                      [0.7916667, 0.04166667],
                      [0.7916667, 0.04166667],
                      [0.875, 0.04166667],
                      [0.875, 0.04166667],
                      [0.875, 0.04166667],
                      [0.875, 0.04166667],
                      [0.875, 0.04166667],
                      [0.875, 0.04166667],
                      [0.9583333, 0.04166667],
                      [0.9583333, 0.04166667],
                      [0.9583333, 0.04166667],
                      [0.9583333, 0.04166667],
                      [0.9583333, 0.04166667],
                      [0.9583333, 0.04166667],
                      [0.04166667, 0.125],
                      [0.04166667, 0.125],
                      [0.04166667, 0.125],
                      [0.04166667, 0.125],
                      [0.04166667, 0.125],
                      [0.04166667, 0.125],
                      [0.125, 0.125],
                      [0.125, 0.125],
                      [0.125, 0.125],
                      [0.125, 0.125],
                      [0.125, 0.125],
                      [0.125, 0.125],
                      [0.20833333, 0.125],
                      [0.20833333, 0.125],
                      [0.20833333, 0.125],
                      [0.20833333, 0.125],
                      [0.20833333, 0.125],
                      [0.20833333, 0.125],
                      [0.29166666, 0.125],
                      [0.29166666, 0.125],
                      [0.29166666, 0.125],
                      [0.29166666, 0.125],
                      [0.29166666, 0.125],
                      [0.29166666, 0.125],
                      [0.375, 0.125],
                      [0.375, 0.125],
                      [0.375, 0.125],
                      [0.375, 0.125],
                      [0.375, 0.125],
                      [0.375, 0.125],
                      [0.45833334, 0.125],
                      [0.45833334, 0.125],
                      [0.45833334, 0.125],
                      [0.45833334, 0.125],
                      [0.45833334, 0.125],
                      [0.45833334, 0.125],
                      [0.5416667, 0.125],
                      [0.5416667, 0.125],
                      [0.5416667, 0.125],
                      [0.5416667, 0.125],
                      [0.5416667, 0.125],
                      [0.5416667, 0.125],
                      [0.625, 0.125],
                      [0.625, 0.125],
                      [0.625, 0.125],
                      [0.625, 0.125],
                      [0.625, 0.125],
                      [0.625, 0.125],
                      [0.7083333, 0.125],
                      [0.7083333, 0.125],
                      [0.7083333, 0.125],
                      [0.7083333, 0.125],
                      [0.7083333, 0.125],
                      [0.7083333, 0.125],
                      [0.7916667, 0.125],
                      [0.7916667, 0.125],
                      [0.7916667, 0.125],
                      [0.7916667, 0.125],
                      [0.7916667, 0.125],
                      [0.7916667, 0.125],
                      [0.875, 0.125],
                      [0.875, 0.125],
                      [0.875, 0.125],
                      [0.875, 0.125],
                      [0.875, 0.125],
                      [0.875, 0.125],
                      [0.9583333, 0.125],
                      [0.9583333, 0.125],
                      [0.9583333, 0.125],
                      [0.9583333, 0.125],
                      [0.9583333, 0.125],
                      [0.9583333, 0.125],
                      [0.04166667, 0.20833333],
                      [0.04166667, 0.20833333],
                      [0.04166667, 0.20833333],
                      [0.04166667, 0.20833333],
                      [0.04166667, 0.20833333],
                      [0.04166667, 0.20833333],
                      [0.125, 0.20833333],
                      [0.125, 0.20833333],
                      [0.125, 0.20833333],
                      [0.125, 0.20833333],
                      [0.125, 0.20833333],
                      [0.125, 0.20833333],
                      [0.20833333, 0.20833333],
                      [0.20833333, 0.20833333],
                      [0.20833333, 0.20833333],
                      [0.20833333, 0.20833333],
                      [0.20833333, 0.20833333],
                      [0.20833333, 0.20833333],
                      [0.29166666, 0.20833333],
                      [0.29166666, 0.20833333],
                      [0.29166666, 0.20833333],
                      [0.29166666, 0.20833333],
                      [0.29166666, 0.20833333],
                      [0.29166666, 0.20833333],
                      [0.375, 0.20833333],
                      [0.375, 0.20833333],
                      [0.375, 0.20833333],
                      [0.375, 0.20833333],
                      [0.375, 0.20833333],
                      [0.375, 0.20833333],
                      [0.45833334, 0.20833333],
                      [0.45833334, 0.20833333],
                      [0.45833334, 0.20833333],
                      [0.45833334, 0.20833333],
                      [0.45833334, 0.20833333],
                      [0.45833334, 0.20833333],
                      [0.5416667, 0.20833333],
                      [0.5416667, 0.20833333],
                      [0.5416667, 0.20833333],
                      [0.5416667, 0.20833333],
                      [0.5416667, 0.20833333],
                      [0.5416667, 0.20833333],
                      [0.625, 0.20833333],
                      [0.625, 0.20833333],
                      [0.625, 0.20833333],
                      [0.625, 0.20833333],
                      [0.625, 0.20833333],
                      [0.625, 0.20833333],
                      [0.7083333, 0.20833333],
                      [0.7083333, 0.20833333],
                      [0.7083333, 0.20833333],
                      [0.7083333, 0.20833333],
                      [0.7083333, 0.20833333],
                      [0.7083333, 0.20833333],
                      [0.7916667, 0.20833333],
                      [0.7916667, 0.20833333],
                      [0.7916667, 0.20833333],
                      [0.7916667, 0.20833333],
                      [0.7916667, 0.20833333],
                      [0.7916667, 0.20833333],
                      [0.875, 0.20833333],
                      [0.875, 0.20833333],
                      [0.875, 0.20833333],
                      [0.875, 0.20833333],
                      [0.875, 0.20833333],
                      [0.875, 0.20833333],
                      [0.9583333, 0.20833333],
                      [0.9583333, 0.20833333],
                      [0.9583333, 0.20833333],
                      [0.9583333, 0.20833333],
                      [0.9583333, 0.20833333],
                      [0.9583333, 0.20833333],
                      [0.04166667, 0.29166666],
                      [0.04166667, 0.29166666],
                      [0.04166667, 0.29166666],
                      [0.04166667, 0.29166666],
                      [0.04166667, 0.29166666],
                      [0.04166667, 0.29166666],
                      [0.125, 0.29166666],
                      [0.125, 0.29166666],
                      [0.125, 0.29166666],
                      [0.125, 0.29166666],
                      [0.125, 0.29166666],
                      [0.125, 0.29166666],
                      [0.20833333, 0.29166666],
                      [0.20833333, 0.29166666],
                      [0.20833333, 0.29166666],
                      [0.20833333, 0.29166666],
                      [0.20833333, 0.29166666],
                      [0.20833333, 0.29166666],
                      [0.29166666, 0.29166666],
                      [0.29166666, 0.29166666],
                      [0.29166666, 0.29166666],
                      [0.29166666, 0.29166666],
                      [0.29166666, 0.29166666],
                      [0.29166666, 0.29166666],
                      [0.375, 0.29166666],
                      [0.375, 0.29166666],
                      [0.375, 0.29166666],
                      [0.375, 0.29166666],
                      [0.375, 0.29166666],
                      [0.375, 0.29166666],
                      [0.45833334, 0.29166666],
                      [0.45833334, 0.29166666],
                      [0.45833334, 0.29166666],
                      [0.45833334, 0.29166666],
                      [0.45833334, 0.29166666],
                      [0.45833334, 0.29166666],
                      [0.5416667, 0.29166666],
                      [0.5416667, 0.29166666],
                      [0.5416667, 0.29166666],
                      [0.5416667, 0.29166666],
                      [0.5416667, 0.29166666],
                      [0.5416667, 0.29166666],
                      [0.625, 0.29166666],
                      [0.625, 0.29166666],
                      [0.625, 0.29166666],
                      [0.625, 0.29166666],
                      [0.625, 0.29166666],
                      [0.625, 0.29166666],
                      [0.7083333, 0.29166666],
                      [0.7083333, 0.29166666],
                      [0.7083333, 0.29166666],
                      [0.7083333, 0.29166666],
                      [0.7083333, 0.29166666],
                      [0.7083333, 0.29166666],
                      [0.7916667, 0.29166666],
                      [0.7916667, 0.29166666],
                      [0.7916667, 0.29166666],
                      [0.7916667, 0.29166666],
                      [0.7916667, 0.29166666],
                      [0.7916667, 0.29166666],
                      [0.875, 0.29166666],
                      [0.875, 0.29166666],
                      [0.875, 0.29166666],
                      [0.875, 0.29166666],
                      [0.875, 0.29166666],
                      [0.875, 0.29166666],
                      [0.9583333, 0.29166666],
                      [0.9583333, 0.29166666],
                      [0.9583333, 0.29166666],
                      [0.9583333, 0.29166666],
                      [0.9583333, 0.29166666],
                      [0.9583333, 0.29166666],
                      [0.04166667, 0.375],
                      [0.04166667, 0.375],
                      [0.04166667, 0.375],
                      [0.04166667, 0.375],
                      [0.04166667, 0.375],
                      [0.04166667, 0.375],
                      [0.125, 0.375],
                      [0.125, 0.375],
                      [0.125, 0.375],
                      [0.125, 0.375],
                      [0.125, 0.375],
                      [0.125, 0.375],
                      [0.20833333, 0.375],
                      [0.20833333, 0.375],
                      [0.20833333, 0.375],
                      [0.20833333, 0.375],
                      [0.20833333, 0.375],
                      [0.20833333, 0.375],
                      [0.29166666, 0.375],
                      [0.29166666, 0.375],
                      [0.29166666, 0.375],
                      [0.29166666, 0.375],
                      [0.29166666, 0.375],
                      [0.29166666, 0.375],
                      [0.375, 0.375],
                      [0.375, 0.375],
                      [0.375, 0.375],
                      [0.375, 0.375],
                      [0.375, 0.375],
                      [0.375, 0.375],
                      [0.45833334, 0.375],
                      [0.45833334, 0.375],
                      [0.45833334, 0.375],
                      [0.45833334, 0.375],
                      [0.45833334, 0.375],
                      [0.45833334, 0.375],
                      [0.5416667, 0.375],
                      [0.5416667, 0.375],
                      [0.5416667, 0.375],
                      [0.5416667, 0.375],
                      [0.5416667, 0.375],
                      [0.5416667, 0.375],
                      [0.625, 0.375],
                      [0.625, 0.375],
                      [0.625, 0.375],
                      [0.625, 0.375],
                      [0.625, 0.375],
                      [0.625, 0.375],
                      [0.7083333, 0.375],
                      [0.7083333, 0.375],
                      [0.7083333, 0.375],
                      [0.7083333, 0.375],
                      [0.7083333, 0.375],
                      [0.7083333, 0.375],
                      [0.7916667, 0.375],
                      [0.7916667, 0.375],
                      [0.7916667, 0.375],
                      [0.7916667, 0.375],
                      [0.7916667, 0.375],
                      [0.7916667, 0.375],
                      [0.875, 0.375],
                      [0.875, 0.375],
                      [0.875, 0.375],
                      [0.875, 0.375],
                      [0.875, 0.375],
                      [0.875, 0.375],
                      [0.9583333, 0.375],
                      [0.9583333, 0.375],
                      [0.9583333, 0.375],
                      [0.9583333, 0.375],
                      [0.9583333, 0.375],
                      [0.9583333, 0.375],
                      [0.04166667, 0.45833334],
                      [0.04166667, 0.45833334],
                      [0.04166667, 0.45833334],
                      [0.04166667, 0.45833334],
                      [0.04166667, 0.45833334],
                      [0.04166667, 0.45833334],
                      [0.125, 0.45833334],
                      [0.125, 0.45833334],
                      [0.125, 0.45833334],
                      [0.125, 0.45833334],
                      [0.125, 0.45833334],
                      [0.125, 0.45833334],
                      [0.20833333, 0.45833334],
                      [0.20833333, 0.45833334],
                      [0.20833333, 0.45833334],
                      [0.20833333, 0.45833334],
                      [0.20833333, 0.45833334],
                      [0.20833333, 0.45833334],
                      [0.29166666, 0.45833334],
                      [0.29166666, 0.45833334],
                      [0.29166666, 0.45833334],
                      [0.29166666, 0.45833334],
                      [0.29166666, 0.45833334],
                      [0.29166666, 0.45833334],
                      [0.375, 0.45833334],
                      [0.375, 0.45833334],
                      [0.375, 0.45833334],
                      [0.375, 0.45833334],
                      [0.375, 0.45833334],
                      [0.375, 0.45833334],
                      [0.45833334, 0.45833334],
                      [0.45833334, 0.45833334],
                      [0.45833334, 0.45833334],
                      [0.45833334, 0.45833334],
                      [0.45833334, 0.45833334],
                      [0.45833334, 0.45833334],
                      [0.5416667, 0.45833334],
                      [0.5416667, 0.45833334],
                      [0.5416667, 0.45833334],
                      [0.5416667, 0.45833334],
                      [0.5416667, 0.45833334],
                      [0.5416667, 0.45833334],
                      [0.625, 0.45833334],
                      [0.625, 0.45833334],
                      [0.625, 0.45833334],
                      [0.625, 0.45833334],
                      [0.625, 0.45833334],
                      [0.625, 0.45833334],
                      [0.7083333, 0.45833334],
                      [0.7083333, 0.45833334],
                      [0.7083333, 0.45833334],
                      [0.7083333, 0.45833334],
                      [0.7083333, 0.45833334],
                      [0.7083333, 0.45833334],
                      [0.7916667, 0.45833334],
                      [0.7916667, 0.45833334],
                      [0.7916667, 0.45833334],
                      [0.7916667, 0.45833334],
                      [0.7916667, 0.45833334],
                      [0.7916667, 0.45833334],
                      [0.875, 0.45833334],
                      [0.875, 0.45833334],
                      [0.875, 0.45833334],
                      [0.875, 0.45833334],
                      [0.875, 0.45833334],
                      [0.875, 0.45833334],
                      [0.9583333, 0.45833334],
                      [0.9583333, 0.45833334],
                      [0.9583333, 0.45833334],
                      [0.9583333, 0.45833334],
                      [0.9583333, 0.45833334],
                      [0.9583333, 0.45833334],
                      [0.04166667, 0.5416667],
                      [0.04166667, 0.5416667],
                      [0.04166667, 0.5416667],
                      [0.04166667, 0.5416667],
                      [0.04166667, 0.5416667],
                      [0.04166667, 0.5416667],
                      [0.125, 0.5416667],
                      [0.125, 0.5416667],
                      [0.125, 0.5416667],
                      [0.125, 0.5416667],
                      [0.125, 0.5416667],
                      [0.125, 0.5416667],
                      [0.20833333, 0.5416667],
                      [0.20833333, 0.5416667],
                      [0.20833333, 0.5416667],
                      [0.20833333, 0.5416667],
                      [0.20833333, 0.5416667],
                      [0.20833333, 0.5416667],
                      [0.29166666, 0.5416667],
                      [0.29166666, 0.5416667],
                      [0.29166666, 0.5416667],
                      [0.29166666, 0.5416667],
                      [0.29166666, 0.5416667],
                      [0.29166666, 0.5416667],
                      [0.375, 0.5416667],
                      [0.375, 0.5416667],
                      [0.375, 0.5416667],
                      [0.375, 0.5416667],
                      [0.375, 0.5416667],
                      [0.375, 0.5416667],
                      [0.45833334, 0.5416667],
                      [0.45833334, 0.5416667],
                      [0.45833334, 0.5416667],
                      [0.45833334, 0.5416667],
                      [0.45833334, 0.5416667],
                      [0.45833334, 0.5416667],
                      [0.5416667, 0.5416667],
                      [0.5416667, 0.5416667],
                      [0.5416667, 0.5416667],
                      [0.5416667, 0.5416667],
                      [0.5416667, 0.5416667],
                      [0.5416667, 0.5416667],
                      [0.625, 0.5416667],
                      [0.625, 0.5416667],
                      [0.625, 0.5416667],
                      [0.625, 0.5416667],
                      [0.625, 0.5416667],
                      [0.625, 0.5416667],
                      [0.7083333, 0.5416667],
                      [0.7083333, 0.5416667],
                      [0.7083333, 0.5416667],
                      [0.7083333, 0.5416667],
                      [0.7083333, 0.5416667],
                      [0.7083333, 0.5416667],
                      [0.7916667, 0.5416667],
                      [0.7916667, 0.5416667],
                      [0.7916667, 0.5416667],
                      [0.7916667, 0.5416667],
                      [0.7916667, 0.5416667],
                      [0.7916667, 0.5416667],
                      [0.875, 0.5416667],
                      [0.875, 0.5416667],
                      [0.875, 0.5416667],
                      [0.875, 0.5416667],
                      [0.875, 0.5416667],
                      [0.875, 0.5416667],
                      [0.9583333, 0.5416667],
                      [0.9583333, 0.5416667],
                      [0.9583333, 0.5416667],
                      [0.9583333, 0.5416667],
                      [0.9583333, 0.5416667],
                      [0.9583333, 0.5416667],
                      [0.04166667, 0.625],
                      [0.04166667, 0.625],
                      [0.04166667, 0.625],
                      [0.04166667, 0.625],
                      [0.04166667, 0.625],
                      [0.04166667, 0.625],
                      [0.125, 0.625],
                      [0.125, 0.625],
                      [0.125, 0.625],
                      [0.125, 0.625],
                      [0.125, 0.625],
                      [0.125, 0.625],
                      [0.20833333, 0.625],
                      [0.20833333, 0.625],
                      [0.20833333, 0.625],
                      [0.20833333, 0.625],
                      [0.20833333, 0.625],
                      [0.20833333, 0.625],
                      [0.29166666, 0.625],
                      [0.29166666, 0.625],
                      [0.29166666, 0.625],
                      [0.29166666, 0.625],
                      [0.29166666, 0.625],
                      [0.29166666, 0.625],
                      [0.375, 0.625],
                      [0.375, 0.625],
                      [0.375, 0.625],
                      [0.375, 0.625],
                      [0.375, 0.625],
                      [0.375, 0.625],
                      [0.45833334, 0.625],
                      [0.45833334, 0.625],
                      [0.45833334, 0.625],
                      [0.45833334, 0.625],
                      [0.45833334, 0.625],
                      [0.45833334, 0.625],
                      [0.5416667, 0.625],
                      [0.5416667, 0.625],
                      [0.5416667, 0.625],
                      [0.5416667, 0.625],
                      [0.5416667, 0.625],
                      [0.5416667, 0.625],
                      [0.625, 0.625],
                      [0.625, 0.625],
                      [0.625, 0.625],
                      [0.625, 0.625],
                      [0.625, 0.625],
                      [0.625, 0.625],
                      [0.7083333, 0.625],
                      [0.7083333, 0.625],
                      [0.7083333, 0.625],
                      [0.7083333, 0.625],
                      [0.7083333, 0.625],
                      [0.7083333, 0.625],
                      [0.7916667, 0.625],
                      [0.7916667, 0.625],
                      [0.7916667, 0.625],
                      [0.7916667, 0.625],
                      [0.7916667, 0.625],
                      [0.7916667, 0.625],
                      [0.875, 0.625],
                      [0.875, 0.625],
                      [0.875, 0.625],
                      [0.875, 0.625],
                      [0.875, 0.625],
                      [0.875, 0.625],
                      [0.9583333, 0.625],
                      [0.9583333, 0.625],
                      [0.9583333, 0.625],
                      [0.9583333, 0.625],
                      [0.9583333, 0.625],
                      [0.9583333, 0.625],
                      [0.04166667, 0.7083333],
                      [0.04166667, 0.7083333],
                      [0.04166667, 0.7083333],
                      [0.04166667, 0.7083333],
                      [0.04166667, 0.7083333],
                      [0.04166667, 0.7083333],
                      [0.125, 0.7083333],
                      [0.125, 0.7083333],
                      [0.125, 0.7083333],
                      [0.125, 0.7083333],
                      [0.125, 0.7083333],
                      [0.125, 0.7083333],
                      [0.20833333, 0.7083333],
                      [0.20833333, 0.7083333],
                      [0.20833333, 0.7083333],
                      [0.20833333, 0.7083333],
                      [0.20833333, 0.7083333],
                      [0.20833333, 0.7083333],
                      [0.29166666, 0.7083333],
                      [0.29166666, 0.7083333],
                      [0.29166666, 0.7083333],
                      [0.29166666, 0.7083333],
                      [0.29166666, 0.7083333],
                      [0.29166666, 0.7083333],
                      [0.375, 0.7083333],
                      [0.375, 0.7083333],
                      [0.375, 0.7083333],
                      [0.375, 0.7083333],
                      [0.375, 0.7083333],
                      [0.375, 0.7083333],
                      [0.45833334, 0.7083333],
                      [0.45833334, 0.7083333],
                      [0.45833334, 0.7083333],
                      [0.45833334, 0.7083333],
                      [0.45833334, 0.7083333],
                      [0.45833334, 0.7083333],
                      [0.5416667, 0.7083333],
                      [0.5416667, 0.7083333],
                      [0.5416667, 0.7083333],
                      [0.5416667, 0.7083333],
                      [0.5416667, 0.7083333],
                      [0.5416667, 0.7083333],
                      [0.625, 0.7083333],
                      [0.625, 0.7083333],
                      [0.625, 0.7083333],
                      [0.625, 0.7083333],
                      [0.625, 0.7083333],
                      [0.625, 0.7083333],
                      [0.7083333, 0.7083333],
                      [0.7083333, 0.7083333],
                      [0.7083333, 0.7083333],
                      [0.7083333, 0.7083333],
                      [0.7083333, 0.7083333],
                      [0.7083333, 0.7083333],
                      [0.7916667, 0.7083333],
                      [0.7916667, 0.7083333],
                      [0.7916667, 0.7083333],
                      [0.7916667, 0.7083333],
                      [0.7916667, 0.7083333],
                      [0.7916667, 0.7083333],
                      [0.875, 0.7083333],
                      [0.875, 0.7083333],
                      [0.875, 0.7083333],
                      [0.875, 0.7083333],
                      [0.875, 0.7083333],
                      [0.875, 0.7083333],
                      [0.9583333, 0.7083333],
                      [0.9583333, 0.7083333],
                      [0.9583333, 0.7083333],
                      [0.9583333, 0.7083333],
                      [0.9583333, 0.7083333],
                      [0.9583333, 0.7083333],
                      [0.04166667, 0.7916667],
                      [0.04166667, 0.7916667],
                      [0.04166667, 0.7916667],
                      [0.04166667, 0.7916667],
                      [0.04166667, 0.7916667],
                      [0.04166667, 0.7916667],
                      [0.125, 0.7916667],
                      [0.125, 0.7916667],
                      [0.125, 0.7916667],
                      [0.125, 0.7916667],
                      [0.125, 0.7916667],
                      [0.125, 0.7916667],
                      [0.20833333, 0.7916667],
                      [0.20833333, 0.7916667],
                      [0.20833333, 0.7916667],
                      [0.20833333, 0.7916667],
                      [0.20833333, 0.7916667],
                      [0.20833333, 0.7916667],
                      [0.29166666, 0.7916667],
                      [0.29166666, 0.7916667],
                      [0.29166666, 0.7916667],
                      [0.29166666, 0.7916667],
                      [0.29166666, 0.7916667],
                      [0.29166666, 0.7916667],
                      [0.375, 0.7916667],
                      [0.375, 0.7916667],
                      [0.375, 0.7916667],
                      [0.375, 0.7916667],
                      [0.375, 0.7916667],
                      [0.375, 0.7916667],
                      [0.45833334, 0.7916667],
                      [0.45833334, 0.7916667],
                      [0.45833334, 0.7916667],
                      [0.45833334, 0.7916667],
                      [0.45833334, 0.7916667],
                      [0.45833334, 0.7916667],
                      [0.5416667, 0.7916667],
                      [0.5416667, 0.7916667],
                      [0.5416667, 0.7916667],
                      [0.5416667, 0.7916667],
                      [0.5416667, 0.7916667],
                      [0.5416667, 0.7916667],
                      [0.625, 0.7916667],
                      [0.625, 0.7916667],
                      [0.625, 0.7916667],
                      [0.625, 0.7916667],
                      [0.625, 0.7916667],
                      [0.625, 0.7916667],
                      [0.7083333, 0.7916667],
                      [0.7083333, 0.7916667],
                      [0.7083333, 0.7916667],
                      [0.7083333, 0.7916667],
                      [0.7083333, 0.7916667],
                      [0.7083333, 0.7916667],
                      [0.7916667, 0.7916667],
                      [0.7916667, 0.7916667],
                      [0.7916667, 0.7916667],
                      [0.7916667, 0.7916667],
                      [0.7916667, 0.7916667],
                      [0.7916667, 0.7916667],
                      [0.875, 0.7916667],
                      [0.875, 0.7916667],
                      [0.875, 0.7916667],
                      [0.875, 0.7916667],
                      [0.875, 0.7916667],
                      [0.875, 0.7916667],
                      [0.9583333, 0.7916667],
                      [0.9583333, 0.7916667],
                      [0.9583333, 0.7916667],
                      [0.9583333, 0.7916667],
                      [0.9583333, 0.7916667],
                      [0.9583333, 0.7916667],
                      [0.04166667, 0.875],
                      [0.04166667, 0.875],
                      [0.04166667, 0.875],
                      [0.04166667, 0.875],
                      [0.04166667, 0.875],
                      [0.04166667, 0.875],
                      [0.125, 0.875],
                      [0.125, 0.875],
                      [0.125, 0.875],
                      [0.125, 0.875],
                      [0.125, 0.875],
                      [0.125, 0.875],
                      [0.20833333, 0.875],
                      [0.20833333, 0.875],
                      [0.20833333, 0.875],
                      [0.20833333, 0.875],
                      [0.20833333, 0.875],
                      [0.20833333, 0.875],
                      [0.29166666, 0.875],
                      [0.29166666, 0.875],
                      [0.29166666, 0.875],
                      [0.29166666, 0.875],
                      [0.29166666, 0.875],
                      [0.29166666, 0.875],
                      [0.375, 0.875],
                      [0.375, 0.875],
                      [0.375, 0.875],
                      [0.375, 0.875],
                      [0.375, 0.875],
                      [0.375, 0.875],
                      [0.45833334, 0.875],
                      [0.45833334, 0.875],
                      [0.45833334, 0.875],
                      [0.45833334, 0.875],
                      [0.45833334, 0.875],
                      [0.45833334, 0.875],
                      [0.5416667, 0.875],
                      [0.5416667, 0.875],
                      [0.5416667, 0.875],
                      [0.5416667, 0.875],
                      [0.5416667, 0.875],
                      [0.5416667, 0.875],
                      [0.625, 0.875],
                      [0.625, 0.875],
                      [0.625, 0.875],
                      [0.625, 0.875],
                      [0.625, 0.875],
                      [0.625, 0.875],
                      [0.7083333, 0.875],
                      [0.7083333, 0.875],
                      [0.7083333, 0.875],
                      [0.7083333, 0.875],
                      [0.7083333, 0.875],
                      [0.7083333, 0.875],
                      [0.7916667, 0.875],
                      [0.7916667, 0.875],
                      [0.7916667, 0.875],
                      [0.7916667, 0.875],
                      [0.7916667, 0.875],
                      [0.7916667, 0.875],
                      [0.875, 0.875],
                      [0.875, 0.875],
                      [0.875, 0.875],
                      [0.875, 0.875],
                      [0.875, 0.875],
                      [0.875, 0.875],
                      [0.9583333, 0.875],
                      [0.9583333, 0.875],
                      [0.9583333, 0.875],
                      [0.9583333, 0.875],
                      [0.9583333, 0.875],
                      [0.9583333, 0.875],
                      [0.04166667, 0.9583333],
                      [0.04166667, 0.9583333],
                      [0.04166667, 0.9583333],
                      [0.04166667, 0.9583333],
                      [0.04166667, 0.9583333],
                      [0.04166667, 0.9583333],
                      [0.125, 0.9583333],
                      [0.125, 0.9583333],
                      [0.125, 0.9583333],
                      [0.125, 0.9583333],
                      [0.125, 0.9583333],
                      [0.125, 0.9583333],
                      [0.20833333, 0.9583333],
                      [0.20833333, 0.9583333],
                      [0.20833333, 0.9583333],
                      [0.20833333, 0.9583333],
                      [0.20833333, 0.9583333],
                      [0.20833333, 0.9583333],
                      [0.29166666, 0.9583333],
                      [0.29166666, 0.9583333],
                      [0.29166666, 0.9583333],
                      [0.29166666, 0.9583333],
                      [0.29166666, 0.9583333],
                      [0.29166666, 0.9583333],
                      [0.375, 0.9583333],
                      [0.375, 0.9583333],
                      [0.375, 0.9583333],
                      [0.375, 0.9583333],
                      [0.375, 0.9583333],
                      [0.375, 0.9583333],
                      [0.45833334, 0.9583333],
                      [0.45833334, 0.9583333],
                      [0.45833334, 0.9583333],
                      [0.45833334, 0.9583333],
                      [0.45833334, 0.9583333],
                      [0.45833334, 0.9583333],
                      [0.5416667, 0.9583333],
                      [0.5416667, 0.9583333],
                      [0.5416667, 0.9583333],
                      [0.5416667, 0.9583333],
                      [0.5416667, 0.9583333],
                      [0.5416667, 0.9583333],
                      [0.625, 0.9583333],
                      [0.625, 0.9583333],
                      [0.625, 0.9583333],
                      [0.625, 0.9583333],
                      [0.625, 0.9583333],
                      [0.625, 0.9583333],
                      [0.7083333, 0.9583333],
                      [0.7083333, 0.9583333],
                      [0.7083333, 0.9583333],
                      [0.7083333, 0.9583333],
                      [0.7083333, 0.9583333],
                      [0.7083333, 0.9583333],
                      [0.7916667, 0.9583333],
                      [0.7916667, 0.9583333],
                      [0.7916667, 0.9583333],
                      [0.7916667, 0.9583333],
                      [0.7916667, 0.9583333],
                      [0.7916667, 0.9583333],
                      [0.875, 0.9583333],
                      [0.875, 0.9583333],
                      [0.875, 0.9583333],
                      [0.875, 0.9583333],
                      [0.875, 0.9583333],
                      [0.875, 0.9583333],
                      [0.9583333, 0.9583333],
                      [0.9583333, 0.9583333],
                      [0.9583333, 0.9583333],
                      [0.9583333, 0.9583333],
                      [0.9583333, 0.9583333],
                      [0.9583333, 0.9583333]], dtype=np.float32)

def visualize(image, hands, print_result=False):
    display_screen = image.copy()
    display_3d = np.zeros((400, 400, 3), np.uint8)
    cv.line(display_3d, (200, 0), (200, 400), (255, 255, 255), 2)
    cv.line(display_3d, (0, 200), (400, 200), (255, 255, 255), 2)
    cv.putText(display_3d, 'Main View', (0, 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Top View', (200, 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Left View', (0, 212), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Right View', (200, 212), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    is_draw = False  # ensure only one hand is drawn

    def draw_lines(image, landmarks, is_draw_point=True, thickness=2):
        cv.line(image, landmarks[0], landmarks[1], (255, 255, 255), thickness)
        cv.line(image, landmarks[1], landmarks[2], (255, 255, 255), thickness)
        cv.line(image, landmarks[2], landmarks[3], (255, 255, 255), thickness)
        cv.line(image, landmarks[3], landmarks[4], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[5], (255, 255, 255), thickness)
        cv.line(image, landmarks[5], landmarks[6], (255, 255, 255), thickness)
        cv.line(image, landmarks[6], landmarks[7], (255, 255, 255), thickness)
        cv.line(image, landmarks[7], landmarks[8], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[9], (255, 255, 255), thickness)
        cv.line(image, landmarks[9], landmarks[10], (255, 255, 255), thickness)
        cv.line(image, landmarks[10], landmarks[11], (255, 255, 255), thickness)
        cv.line(image, landmarks[11], landmarks[12], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[13], (255, 255, 255), thickness)
        cv.line(image, landmarks[13], landmarks[14], (255, 255, 255), thickness)
        cv.line(image, landmarks[14], landmarks[15], (255, 255, 255), thickness)
        cv.line(image, landmarks[15], landmarks[16], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[17], (255, 255, 255), thickness)
        cv.line(image, landmarks[17], landmarks[18], (255, 255, 255), thickness)
        cv.line(image, landmarks[18], landmarks[19], (255, 255, 255), thickness)
        cv.line(image, landmarks[19], landmarks[20], (255, 255, 255), thickness)

        if is_draw_point:
            for p in landmarks:
                cv.circle(image, p, thickness, (0, 0, 255), -1)

    # used for gesture classification
    gc = GestureClassification()

    for idx, handpose in enumerate(hands):
        conf = handpose[-1]
        bbox = handpose[0:4].astype(np.int32)
        handedness = handpose[-2]
        if handedness <= 0.5:
            handedness_text = 'Left'
        else:
            handedness_text = 'Right'
        landmarks_screen = handpose[4:67].reshape(21, 3).astype(np.int32)
        landmarks_word = handpose[67:130].reshape(21, 3)

        gesture = gc.classify(landmarks_screen)

        # Print results
        if print_result:
            print('-----------hand {}-----------'.format(idx + 1))
            print('conf: {:.2f}'.format(conf))
            print('handedness: {}'.format(handedness_text))
            print('gesture: {}'.format(gesture))
            print('hand box: {}'.format(bbox))
            print('hand landmarks: ')
            for l in landmarks_screen:
                print('\t{}'.format(l))
            print('hand world landmarks: ')
            for l in landmarks_word:
                print('\t{}'.format(l))

        # draw box
        cv.rectangle(display_screen, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # draw handedness
        cv.putText(display_screen, '{}'.format(handedness_text), (bbox[0], bbox[1] + 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        # draw gesture
        cv.putText(display_screen, '{}'.format(gesture), (bbox[0], bbox[1] + 30), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        # Draw line between each key points
        landmarks_xy = landmarks_screen[:, 0:2]
        draw_lines(display_screen, landmarks_xy, is_draw_point=False)

        # z value is relative to WRIST
        for p in landmarks_screen:
            r = max(5 - p[2] // 5, 0)
            r = min(r, 14)
            cv.circle(display_screen, np.array([p[0], p[1]]), r, (0, 0, 255), -1)

        if is_draw is False:
            is_draw = True
            # Main view
            landmarks_xy = landmarks_word[:, [0, 1]]
            landmarks_xy = (landmarks_xy * 1000 + 100).astype(np.int32)
            draw_lines(display_3d, landmarks_xy, thickness=5)

            # Top view
            landmarks_xz = landmarks_word[:, [0, 2]]
            landmarks_xz[:, 1] = -landmarks_xz[:, 1]
            landmarks_xz = (landmarks_xz * 1000 + np.array([300, 100])).astype(np.int32)
            draw_lines(display_3d, landmarks_xz, thickness=5)

            # Left view
            landmarks_yz = landmarks_word[:, [2, 1]]
            landmarks_yz[:, 0] = -landmarks_yz[:, 0]
            landmarks_yz = (landmarks_yz * 1000 + np.array([100, 300])).astype(np.int32)
            draw_lines(display_3d, landmarks_yz, thickness=5)

            # Right view
            landmarks_zy = landmarks_word[:, [2, 1]]
            landmarks_zy = (landmarks_zy * 1000 + np.array([300, 300])).astype(np.int32)
            draw_lines(display_3d, landmarks_zy, thickness=5)

    return display_screen, display_3d

class GestureClassification:
    def _vector_2_angle(self, v1, v2):
        uv1 = v1 / np.linalg.norm(v1)
        uv2 = v2 / np.linalg.norm(v2)
        angle = np.degrees(np.arccos(np.dot(uv1, uv2)))
        return angle

    def _hand_angle(self, hand):
        angle_list = []
        # thumb
        angle_ = self._vector_2_angle(
            np.array([hand[0][0] - hand[2][0], hand[0][1] - hand[2][1]]),
            np.array([hand[3][0] - hand[4][0], hand[3][1] - hand[4][1]])
        )
        angle_list.append(angle_)
        # index
        angle_ = self._vector_2_angle(
            np.array([hand[0][0] - hand[6][0], hand[0][1] - hand[6][1]]),
            np.array([hand[7][0] - hand[8][0], hand[7][1] - hand[8][1]])
        )
        angle_list.append(angle_)
        # middle
        angle_ = self._vector_2_angle(
            np.array([hand[0][0] - hand[10][0], hand[0][1] - hand[10][1]]),
            np.array([hand[11][0] - hand[12][0], hand[11][1] - hand[12][1]])
        )
        angle_list.append(angle_)
        # ring
        angle_ = self._vector_2_angle(
            np.array([hand[0][0] - hand[14][0], hand[0][1] - hand[14][1]]),
            np.array([hand[15][0] - hand[16][0], hand[15][1] - hand[16][1]])
        )
        angle_list.append(angle_)
        # pink
        angle_ = self._vector_2_angle(
            np.array([hand[0][0] - hand[18][0], hand[0][1] - hand[18][1]]),
            np.array([hand[19][0] - hand[20][0], hand[19][1] - hand[20][1]])
        )
        angle_list.append(angle_)
        return angle_list

    def _finger_status(self, lmList):
        fingerList = []
        originx, originy = lmList[0]
        keypoint_list = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
        for point in keypoint_list:
            x1, y1 = lmList[point[0]]
            x2, y2 = lmList[point[1]]
            if np.hypot(x2 - originx, y2 - originy) > np.hypot(x1 - originx, y1 - originy):
                fingerList.append(True)
            else:
                fingerList.append(False)

        return fingerList

    def _classify(self, hand):
        thr_angle = 65.
        thr_angle_thumb = 30.
        thr_angle_s = 49.
        gesture_str = "Undefined"

        angle_list = self._hand_angle(hand)

        thumbOpen, firstOpen, secondOpen, thirdOpen, fourthOpen = self._finger_status(hand)
        # Number
        if (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                not firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Zero"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "One"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                not thumbOpen and firstOpen and secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Two"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle) and \
                not thumbOpen and firstOpen and secondOpen and thirdOpen and not fourthOpen:
            gesture_str = "Three"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle) and \
                firstOpen and secondOpen and thirdOpen and fourthOpen:
            gesture_str = "Four"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and thirdOpen and fourthOpen:
            gesture_str = "Five"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s) and \
                thumbOpen and not firstOpen and not secondOpen and not thirdOpen and fourthOpen:
            gesture_str = "Six"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Seven"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] < thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Eight"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] < thr_angle) and (
                angle_list[3] < thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and thirdOpen and not fourthOpen:
            gesture_str = "Nine"

        return gesture_str

    def classify(self, landmarks):
        hand = landmarks[:21, :2]
        gesture = self._classify(hand)
        return gesture