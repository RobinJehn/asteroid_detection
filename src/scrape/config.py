import os

# DB
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "StarShooter"

# Requests
BASE = "https://totas.cosmos.esa.int/"

# File paths
DATA_FOLDER = os.path.abspath("./../../data")
IMAGE_FOLDER = f"{DATA_FOLDER}/images"
COA_IMAGE_FOLDER = f"{IMAGE_FOLDER}/centered_on_asteroid/"
