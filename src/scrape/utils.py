import os
import subprocess
import time
import requests

from bs4 import BeautifulSoup
from typing import List, Tuple
from mysql.connector.cursor import MySQLCursorAbstract
from mysql.connector.connection import MySQLConnection

from db import insert_into_table, create_image_metadata_table, create_position_table


def extract_sohas_designation_tag(soup: BeautifulSoup) -> Tuple[str, str]:
    """
    Extract sohas_designation and classification tag from the html,
    e.g. "Mover OG11126 - real asteroid" -> ("OG11126", "real asteroid")

    Args:
        soup (BeautifulSoup): The soup of the page

    Returns: Tuple of sohas_designation and classification tag
    """
    cleaned_str = (
        str(soup.findAll("h3")[-1])
        .replace("<h3>", "")
        .replace("</h3>", "")
        .replace("Mover ", "")
        .replace("\n", "")
    )
    if " - " not in cleaned_str:
        return cleaned_str, ""
    sohas_designation, tag = cleaned_str.split(" - ", maxsplit=1)
    return sohas_designation, tag


def get_centered_on_asteroid_image_links(soup: BeautifulSoup) -> List[str]:
    """
    Get the links to the images that are centered on the asteroid from the html

    Args:
        soup (BeautifulSoup): The soup of the page

    Returns: List of image links
    """
    # Get the index of the image row
    contents = soup.findAll("tr")
    image_row_idx = -1
    for idx, content in enumerate(contents):
        if content.find("td").find("br"):
            image_row_idx = idx + 1
            break
    image_links = [img["src"] for img in contents[image_row_idx].findAll("img")[1:]]
    return image_links


def download_images(
    image_links: List[str],
    base: str,
    output_dir: str,
    cursor: MySQLCursorAbstract,
    sohas_designation: str,
    sleep: int = 1,
) -> None:
    """
    Download the images and save the information to the database

    Args:
        image_links (List[str]): The links to the images
        base (str): The base url
        output_dir (str): The directory to save the images
        cursor (MySQLCursorAbstract): The cursor to the database
        sleep (int, optional): Time to sleep between downloads. Defaults to 1.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image_link in image_links:
        image_name = image_link.split("/")[-1]
        subprocess.run(
            f"wget {base + image_link} -O {output_dir}/{image_name}",
            shell=True,
            capture_output=True,
        )
        insert_into_table(
            cursor, "centered_image", [image_name, image_link, sohas_designation]
        )
        time.sleep(sleep)


def extract_image_data(
    soup: BeautifulSoup,
    cursor: MySQLCursorAbstract,
) -> None:
    """
    Extract the meta data of the images froum the soup and store the metadata in the database

    Args:
        soup (BeautifulSoup): The soup of the page
        cursor (MySQLCursorAbstract): The cursor to the database
    """
    rows = soup.findAll("tr")
    meta_data = [
        [
            row.find_all("a")[0]["href"].split("=")[-1],  # image_id
            *[data.text for data in row.find_all("td")[1:]],
        ]
        for row in rows
        if ".fit" in row.text
    ]
    for data in meta_data:
        insert_into_table(cursor, "image_metadata", data)


def extract_image_row_titles(
    soup: BeautifulSoup,
    cursor: MySQLCursorAbstract,
) -> None:
    """
    Extract the titles of the image rows and create the image metadata table

    Args:
        soup (BeautifulSoup): The soup of the page
        cursor (MySQLCursorAbstract): The cursor to the database
    """
    rows = soup.findAll("tr")
    title_row = None
    for row in rows:
        if "FileName" in row.text:
            title_row = row
            break
    create_image_metadata_table(
        cursor,
        [data.text for data in title_row.find_all("th")[1:]],
    )


def extract_position_data(
    soup: BeautifulSoup,
    cursor: MySQLCursorAbstract,
    image_links: List[str],
) -> None:
    """
    Extract the position data from the soup and store it in the database.
    Associate the position data with the image data by assuming they are in the same order.

    Args:
        soup (BeautifulSoup): The soup of the page
        mycursor (MySQLCursorAbstract): The cursor to the database
        image_links (List[str]): The links to the images
    """
    rows = soup.findAll("tr")
    position_data = [
        [
            *[data.text for data in row.find_all("td")[1:]],
        ]
        for row in rows
        if "position" in str(row)
    ]

    for idx, data in enumerate(position_data):
        if idx < len(image_links):
            file_name = image_links[idx].split("/")[-1]
        else:
            file_name = "NOT_FOUND"

        insert_into_table(cursor, "position", [file_name] + data)


def extract_position_row_titles(
    soup: BeautifulSoup,
    cursor: MySQLCursorAbstract,
) -> None:
    """
    Extract the titles of the position rows and create the position table

    Args:
        soup (BeautifulSoup): The soup of the page
        cursor (MySQLCursorAbstract): The cursor to the database
    """
    rows = soup.findAll("tr")
    title_row = None
    for row in rows:
        if "MpcLine" in row.text:
            title_row = row
            break
    create_position_table(
        cursor,
        [data.text for data in title_row.find_all("th")[1:]],
    )


def download_mover_data(
    mover_id: int,
    base: str,
    bad_request_output: str,
    centered_image_folder: str,
    db: MySQLConnection,
    cursor: MySQLCursorAbstract,
    first_time: bool,
    sleep: float = 0.5,
) -> bool:
    """
    Download images of a mover centered on the asteroid and save the meta data of the images

    Args:
        mover_id (int): The mover id, e.g. 123
        base (str): The base url
        bad_request_output (str): The output of the bad request
        centered_image_folder (str): The directory to save the centered on asteroid images
        db (MySQLConnection): The connection to the database
        cursor (MySQLCursorAbstract): The cursor to the database
        first_time (bool): Whether this is the first time downloading the data, i.e. creating the tables
        sleep (int, optional): Time to sleep between downloads. Defaults to 1.

    Returns whether the request was successful or not
    """
    # Get html
    url = base + f"/mover.php?id={mover_id}"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")

    # Whether request was successful
    if str(soup.prettify()) == bad_request_output:
        insert_into_table(cursor, "not_found", [mover_id])
        db.commit()
        return False

    # Get the data
    sohas_designation, tag = extract_sohas_designation_tag(soup)
    image_links = get_centered_on_asteroid_image_links(soup)
    download_images(
        image_links, base, centered_image_folder, cursor, sohas_designation, sleep
    )

    if first_time:
        extract_image_row_titles(soup, cursor)
        extract_position_row_titles(soup, cursor)

    extract_image_data(soup, cursor)
    extract_position_data(soup, cursor, image_links)

    insert_into_table(cursor, "mover", [mover_id, sohas_designation, tag])

    db.commit()
    return True
