import os
import sys

sys.path.append("..")

from dotenv import load_dotenv, find_dotenv
from utils.models import download_and_extract_model, download_labels
from wcod_utils.data_setup import create_directories


def main() -> None:
    # load env vars
    try:
        load_dotenv(find_dotenv())
    except Exception as e:
        print(e)

    DATA_DIR = os.path.join(os.getcwd(), 'data')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')

    create_directories(DATA_DIR, MODELS_DIR)

    download_and_extract_model( MODELS_DIR,
                                os.getenv("MODEL_DATE"),
                                os.getenv("MODEL_NAME"),
                                os.getenv("MODELS_DOWNLOAD_BASE"))
    download_labels( MODELS_DIR,
                     os.getenv("MODEL_NAME"),
                     os.getenv("LABEL_FILENAME"),
                     os.getenv("LABELS_DOWNLOAD_BASE"))


if __name__ == "__main__":
    main()
