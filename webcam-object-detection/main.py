import os
import sys

sys.path.append("..")

from dotenv import load_dotenv, find_dotenv
from utils.models import download_and_extract_model, download_labels, load_model
from wcod_utils.directory_setup import create_directories

@tf.function
def detect_fn(image):
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

    ckpt = load_model(PATH_TO_CFG, PATH_TO_CKPT)

    category_index = label_map_util.create_category_index_from_labelmap(str(PATH_TO_LABELS),
                                                                        use_display_name=True)



if __name__ == "__main__":
    main()
