import json

# Change constant values based on folder and file structure

# `preprocess.py`
RESOURCE_PATH = "./resources"
KERN_DATASET_PATH = "./elsass"
SAVE_DIR = f"./dataset"
MAPPING_PATH = f"./{RESOURCE_PATH}/mapping.json"

SINGLE_FILE_DATASET = f"./{RESOURCE_PATH}/file_dataset"
SEQUENCE_LENGTH = 100  # 64
MUSIC_XML_PATH = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

# `train.py`
try:
    with open(MAPPING_PATH, 'r') as file:
        OUTPUT_UNITS = len(json.load(file))  # `mapping.json` length; length of json - 2
except Exception as e:
    pass

NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"
