from tqdm import tqdm
from db import (
    create_mover_table,
    connect_to_db,
    create_centered_image_table,
    create_not_found_table,
)
from utils import download_mover_data
from config import (
    COA_IMAGE_FOLDER,
    BASE,
)

TOTAL_MOVERS = 600_000
bad_request_output = "<p>\n nice try.\n <br/>\n logged.\n <br/>\n bye.\n</p>\n"

db = connect_to_db()
cursor = db.cursor()

create_mover_table(cursor)
create_centered_image_table(cursor)
create_not_found_table(cursor)

first_time = True
for mover_id in tqdm(range(1, TOTAL_MOVERS + 1)):
    # Check if mover_id is in the table mover of db
    cursor.execute("SELECT COUNT(*) FROM mover WHERE mover_id = %s", (mover_id,))
    result = cursor.fetchone()
    if result[0] > 0:
        continue
    # Check if mover_id is in the table not_found of db
    cursor.execute("SELECT COUNT(*) FROM not_found WHERE mover_id = %s", (mover_id,))
    result = cursor.fetchone()
    if result[0] > 0:
        continue

    download_mover_data(
        mover_id, BASE, bad_request_output, COA_IMAGE_FOLDER, db, cursor, first_time
    )
    first_time = False
