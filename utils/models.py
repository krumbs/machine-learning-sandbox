import os
import tarfile
import urllib.request
from pathlib import Path

import sys
sys.path.append("../models/research")

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

def download_and_extract_model( MODELS_DIR,
                                MODEL_DATE,
                                MODEL_NAME,
                                MODELS_DOWNLOAD_BASE) -> [Path, Path]:

    MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
    MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
    PATH_TO_MODEL_TAR = Path(MODELS_DIR, MODEL_TAR_FILENAME).as_posix()
    PATH_TO_CKPT = Path(MODELS_DIR, MODEL_NAME, 'checkpoint/')
    PATH_TO_CFG = Path(MODELS_DIR, MODEL_NAME, 'pipeline.config')

    # Download and extract model
    if not os.path.exists(PATH_TO_CKPT):
        print('Downloading model. This may take a while... ', end='')
        urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
        tar_file = tarfile.open(PATH_TO_MODEL_TAR)
        tar_file.extractall(MODELS_DIR)
        tar_file.close()
        os.remove(PATH_TO_MODEL_TAR)
        print('Done')
    else:
        print('Model already downloaded.')

    return PATH_TO_CKPT, PATH_TO_CFG

def download_labels(MODELS_DIR,
                    MODEL_NAME,
                    LABEL_FILENAME,
                    LABELS_DOWNLOAD_BASE) -> None:
    # Download labels file
    PATH_TO_LABELS = Path(MODELS_DIR, MODEL_NAME, LABEL_FILENAME)
    # TODO: change this from os.path.exsits(path) to pathlib.Path.exists()
    if not os.path.exists(PATH_TO_LABELS):
        print('Downloading label file... ', end='')
        urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
        print('Done')
    else:
        print('Labels already downloaded.')

    return PATH_TO_LABELS


def load_model(PATH_TO_CKPT, PATH_TO_CFG):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging

    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(str(PATH_TO_CFG))
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(Path(PATH_TO_CKPT, 'ckpt-0').as_posix()).expect_partial()

    return detection_model
