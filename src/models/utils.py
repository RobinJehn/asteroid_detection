from mysql.connector.cursor import MySQLCursorAbstract
from tqdm import tqdm
from typing import Tuple
import cv2
import torch

# Suppress ffmpeg warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ffmpeg")

def get_dataset(cursor: MySQLCursorAbstract, path_to_images: str, image_shape: Tuple[int, int] = (30, 30), images_per_sequence: int = 3):
    query = """SELECT
                    mover.label, 
                    GROUP_CONCAT(centered_image.file_name) AS file_names
                FROM mover
                JOIN centered_image 
                    ON mover.sohas_designation = centered_image.sohas_designation
                JOIN (
                    SELECT sohas_designation
                    FROM mover
                    GROUP BY sohas_designation
                    HAVING COUNT(*) = 1
                ) AS frequent_designations ON mover.sohas_designation = frequent_designations.sohas_designation
                GROUP BY 
                    mover.mover_id, 
                    mover.sohas_designation, 
                    mover.label;"""
    cursor.execute(query)

    x_tensors = []
    y_hat_tensors = []
    for mover in tqdm(cursor.fetchall()):
        label = mover[0]
        file_names = mover[1].split(",")
        if len(file_names) != images_per_sequence:
            continue
        image_tensors = []
        for file_name in file_names:
            image_path = f"{path_to_images}/{file_name}"
            # Check if the file exists
            if os.path.isfile(image_path):
                # Read the image in grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # Check if the image shape matches the expected shape
                if image.shape != image_shape:
                    break
                # Convert the image to a tensor
                image_tensor = torch.tensor(image, dtype=torch.float32)
                # Add batch and channel dimensions (N, C, H, W)
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
                image_tensors.append(image_tensor)
            else:
                break
        else:
            x_tensor = torch.cat(image_tensors, dim=2)
            if x_tensor.shape[2] != 90:
                print(f"Skipping {file_names} because of shape {x_tensor.shape}")
            x_tensors.append(x_tensor)
            y_hat_tensors.append(torch.Tensor([["steroid" in label]]))
    
    x = torch.concat(x_tensors)
    y_hat = torch.concat(y_hat_tensors)

    n_real_asteroids = int(y_hat.sum())
    n_bogus_asteroids = int(y_hat.shape[0] - n_real_asteroids)
    print(
        f"Movers: {n_real_asteroids + n_bogus_asteroids}, Real asteroids: {n_real_asteroids}, Bogus asteroids: {n_bogus_asteroids}"
    )
    print(f"X shape: {x.shape}, Y shape: {y_hat.shape}")


if __name__ == "__main__":
    import sys
    import os

    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the db directory
    db_dir = os.path.join(current_dir, '../db')

    # Insert the db directory into sys.path
    sys.path.insert(0, db_dir)

    from db import connect_to_db
    mydb = connect_to_db()
    mycursor = mydb.cursor()
    get_dataset(mycursor, "../../data/images/centered_on_asteroid")
