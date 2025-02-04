# create folder if it does not exist
import json
import os


def create_folder_if_not_exists(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)


# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)
