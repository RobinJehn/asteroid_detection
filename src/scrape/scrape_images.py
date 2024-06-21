from tqdm import tqdm
import os

from utils import download_images
from db import connect_to_db
from config import COA_IMAGE_FOLDER, BASE

db = connect_to_db()
cursor = db.cursor()

cursor.execute("SELECT * FROM centered_image")

already_downloaded = set(os.listdir(COA_IMAGE_FOLDER))
for row in tqdm(cursor.fetchall()):
    if row[0] in already_downloaded:
        continue
    download_images(
        [row[1]],
        BASE,
        COA_IMAGE_FOLDER,
        cursor,
        row[2],
    )
