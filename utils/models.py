import os
import tarfile
import urllib.request
from pathlib import Path

def download_and_extract_model( MODELS_DIR,
                                MODEL_DATE,
                                MODEL_NAME,
                                MODELS_DOWNLOAD_BASE) -> None:

    MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
    MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
    PATH_TO_MODEL_TAR = Path(MODELS_DIR, MODEL_TAR_FILENAME).as_posix()
    PATH_TO_CKPT = Path(MODELS_DIR, MODEL_NAME, 'checkpoint/').as_posix()
    PATH_TO_CFG = Path(MODELS_DIR, MODEL_NAME, 'pipeline.config').as_posix()

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



def download_labels(MODELS_DIR,
                    MODEL_NAME,
                    LABEL_FILENAME,
                    LABELS_DOWNLOAD_BASE) -> None:
    # Download labels file

    PATH_TO_LABELS = Path(MODELS_DIR, MODEL_NAME, LABEL_FILENAME).as_posix()
    if not os.path.exists(PATH_TO_LABELS):
        print('Downloading label file... ', end='')
        urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
        print('Done')
    else:
        print('Labels already downloaded.')
