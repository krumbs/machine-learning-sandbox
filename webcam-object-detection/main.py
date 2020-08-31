import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

import sys
sys.path.append("../models/research")
sys.path.append("..")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from dotenv import load_dotenv, find_dotenv
from utils.models import download_and_extract_model, download_labels, load_model
from wcod_utils.directory_setup import create_directories

@tf.function
def detect_fn(detection_model, image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

def main() -> None:
    # load env vars
    try:
        load_dotenv(find_dotenv())
    except Exception as e:
        print(e)

    DATA_DIR = os.path.join(os.getcwd(), 'data')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')

    create_directories(DATA_DIR, MODELS_DIR)

    PATH_TO_CFG, PATH_TO_CKPT = download_and_extract_model( MODELS_DIR,
                                                            os.getenv("MODEL_DATE"),
                                                            os.getenv("MODEL_NAME"),
                                                            os.getenv("MODELS_DOWNLOAD_BASE"))
    PATH_TO_LABELS = download_labels(MODELS_DIR,
                                     os.getenv("MODEL_NAME"),
                                     os.getenv("LABEL_FILENAME"),
                                     os.getenv("LABELS_DOWNLOAD_BASE"))

    detection_model = load_model(PATH_TO_CFG, PATH_TO_CKPT)

    category_index = label_map_util.create_category_index_from_labelmap(str(PATH_TO_LABELS),
                                                                        use_display_name=True)

    # Start capturing
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        count+=1
        # Read frame from camera
        ret, image_np = cap.read()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #image_np_expanded = np.expand_dims(image_np, axis=0)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, axis=0), dtype=tf.float32)

        detections, predictions_dict, shapes = detect_fn(detection_model, input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'][0].numpy(),
              (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
              detections['detection_scores'][0].numpy(),
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=.30,
              agnostic_mode=False)

        # Display output
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
        img_path = str(Path('data', f'test_{count}.jpg'))
        cv2.imwrite(img_path, cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if count == 10:
            print('BREAKING DUE TO COUNT')
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
