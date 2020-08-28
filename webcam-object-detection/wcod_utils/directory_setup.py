import os

def create_directories(DATA_DIR, MODELS_DIR) -> None:
    """Creates a directories data/ and models/ if they do not already exist"""


    for dir in [DATA_DIR, MODELS_DIR]:
        if not os.path.exists(dir):
            print(f"Creating {dir}")
            os.mkdir(dir)
        else:
            print("data/ already exists. Skipping.")
