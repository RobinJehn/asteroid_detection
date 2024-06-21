import mysql.connector
from typing import List
from mysql.connector.cursor import MySQLCursorAbstract
from mysql.connector.connection import MySQLConnection

from config import DB_NAME, DB_HOST, DB_USER, DB_PASSWORD


def create_db(
    host: str = DB_HOST,
    user: str = DB_USER,
    password: str = DB_PASSWORD,
    db_name: str = DB_NAME,
):
    """Create a database with the given name if it does not already exist."""
    mydb = mysql.connector.connect(host=host, user=user, password=password)
    mycursor = mydb.cursor()

    mycursor.execute("SHOW DATABASES")

    for x in mycursor:
        if x[0] == db_name:
            print("Database already exists")
            return

    mycursor.execute(f"CREATE DATABASE {db_name}")


def connect_to_db(
    host: str = DB_HOST,
    user: str = DB_USER,
    password: str = DB_PASSWORD,
    database: str = DB_NAME,
) -> MySQLConnection:
    """Connect to the database with the given name."""
    mydb = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    return mydb


def create_table(mycursor: MySQLCursorAbstract, table_name: str, columns: List[str]):
    """Create a table with the given name and columns if table doesn't exist already."""
    mycursor.execute(f"SHOW TABLES LIKE '{table_name}'")
    if mycursor.fetchone():
        print(f"Table {table_name} already exists")
        return

    column_string = ", ".join(columns)
    mycursor.execute(f"CREATE TABLE {table_name} ({column_string})")


def create_mover_table(mycursor: MySQLCursorAbstract):
    """Create the mover table."""
    create_table(
        mycursor,
        "mover",
        [
            "mover_id INT PRIMARY KEY",
            "sohas_designation VARCHAR(255)",
            "label VARCHAR(255)",
        ],
    )


def create_image_metadata_table(mycursor: MySQLCursorAbstract, columns: List[str]):
    """Create the image metadata table.

    Args:
        columns: The columns to be added to the table. Should not include the FileName column.
    """
    all_columns = ["image_id INT PRIMARY KEY"] + [
        f"{column} VARCHAR(255)" for column in columns
    ]

    print(all_columns)
    create_table(
        mycursor,
        "image_metadata",
        all_columns,
    )


def create_position_table(mycursor: MySQLCursorAbstract, columns: List[str]):
    """Create the position table.

    Args:
        columns: The columns to be added to the table. Should not include the MpcLine column.
    """
    all_columns = ["file_name VARCHAR(255) PRIMARY KEY"] + [
        f"{column} VARCHAR(255)" for column in columns
    ]
    create_table(
        mycursor,
        "position",
        all_columns,
    )


def create_centered_image_table(mycursor: MySQLCursorAbstract):
    """Create the centered image table.

    Args:
        mycursor: The cursor to the database.
    """
    create_table(
        mycursor,
        "centered_image",
        [
            "file_name VARCHAR(255) PRIMARY KEY",
            "image_url VARCHAR(255)",
            "sohas_designation VARCHAR(255)",
        ],
    )


def create_not_found_table(mycursor: MySQLCursorAbstract):
    """Create the not found table.

    Args:
        mycursor: The cursor to the database.
    """
    create_table(
        mycursor,
        "not_found",
        ["mover_id INT PRIMARY KEY"],
    )


def insert_into_table(
    mycursor: MySQLCursorAbstract, table_name: str, values: List[str]
):
    """Insert values into the table.

    Args:
        table_name: The name of the table to insert into.
        values: The values to insert into the table.
    """
    value_string = "%s, " * (len(values) - 1) + "%s"
    mycursor.execute(f"INSERT IGNORE INTO {table_name} VALUES ({value_string})", values)


if __name__ == "__main__":
    create_db()
    mydb = connect_to_db()
    mycursor = mydb.cursor()


# Not Found table
# mover_id INT PRIMARY KEY, e.g. 301835 from https://totas.cosmos.esa.int/mover.php?id=301835

# Mover table
# mover_id INT PRIMARY KEY, e.g. 301835 from https://totas.cosmos.esa.int/mover.php?id=301835
# sohas_designation VARCHAR(255), e.g. TO18060
# label VARCHAR(255), e.g. "real asteroid", might be converted to class later

# For this table we don't have which mover it belongs to. We can get that later based on the image_id if we need it
# Image Metadata table
# image_id INT PRIMARY KEY, e.g. 129846 from https://totas.cosmos.esa.int/image.php?id=129846
# meta data ...

# Position table
# sohas_designation VARCHAR(255), e.g. TO18060
# meta data ...

# Centered Image table
# sohas_designation VARCHAR(255) PRIMARY KEY, e.g. TO18060
# image_url VARCHAR(255), e.g. https://totas.cosmos.esa.int/image-view.php?file=2015/11/06/028012/movers/b0301835-4.png
# file_name VARCHAR(255), e.g. b0301835-4.png from https://totas.cosmos.esa.int/image-view.php?file=2015/11/06/028012/movers/b0301835-4.png