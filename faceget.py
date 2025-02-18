#!/usr/bin/env python3
# FaceGet - Version 1.0
# A tool to crawl websites, download images, and detect faces.
# Author: Gill-Bates | https://github.com/Gill-Bates/faceget
# License: MIT
# Date: February 18, 2025

import os
import requests
from bs4 import BeautifulSoup
import cv2
import csv
import re
import argparse
from colorama import init, Fore
import unicodedata
import hashlib
from datetime import datetime
import signal
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from urllib.parse import urljoin, urlparse, urlunparse
from skimage.metrics import structural_similarity as ssim
from threading import Lock
from queue import Queue

# Initialize colorama
init(autoreset=True)

# ====================== CONSTANTS AND CONFIGURATION ======================
# Image processing constants
MIN_IMAGE_SIZE = (300, 300)  # Minimum image dimensions (width, height)
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_DETECTION_MIN_SIZE = (30, 30)
SSIM_THRESHOLD = 0.9  # Threshold for considering images as duplicates

# File and folder constants
ALLOWED_IMAGE_EXTENSIONS = ('.bmp', '.png', '.jpg', '.jpeg', '.webp')
LOG_FILE = 'file.log'
CSV_FILENAME_PREFIX = '_'

# Threading constants
MAX_WORKERS = os.cpu_count() or 4

# ====================== HELPER FUNCTIONS ======================
def setup_logging():
    """Set up logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(LOG_FILE, mode='w')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

# Initialize logging
logger = setup_logging()

# Global variables
crawl_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Signal handler for clean exit
def signal_handler(sig, frame):
    """Handle interrupt signals for a clean exit."""
    logger.info(Fore.LIGHTRED_EX + '\n[ABORT] Interruption detected. Cleaning up and exiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Load the Haar Cascade model
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Lock for thread-safe file operations
file_lock = Lock()

def validate_and_convert_image(image_path):
    """Validate and convert an image to a format suitable for OpenCV processing."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(Fore.YELLOW + f"Unable to read image: {image_path}")
            return None

        # Check if the image has valid dimensions
        if image.size == 0:
            logger.warning(Fore.YELLOW + f"Image {image_path} has size 0")
            return None

        # Convert to 3-channel RGB if necessary
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # RGBA image
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image
    except Exception as e:
        logger.error(Fore.RED + f"[ERROR] Validating and converting image {image_path}: {str(e)}")
        return None

def detect_faces(image_path):
    """Detect faces in an image using Haar Cascade."""
    try:
        image = validate_and_convert_image(image_path)
        if image is None:
            return 0

        if image.shape[0] < MIN_IMAGE_SIZE[1] or image.shape[1] < MIN_IMAGE_SIZE[0]:
            logger.warning(Fore.YELLOW + f"Image {image_path} is too small for face detection.")
            return 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION_SCALE_FACTOR,
            minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
            minSize=FACE_DETECTION_MIN_SIZE
        )
        return len(faces)
    except Exception as e:
        logger.error(Fore.RED + f"[ERROR] Detecting faces in {image_path}: {str(e)}")
        return 0

def safe_filename(filename):
    """Remove invalid characters from filenames."""
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    filename = re.sub(r'[^-\w.]', '', filename)
    return filename

def sha256_checksum(filepath):
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

def create_single_folder_name(base_url):
    """Create a folder name based on the domain of the URL."""
    folder_name = base_url.split("//")[-1].split("/")[0]
    folder_name = unicodedata.normalize('NFKD', folder_name).encode('ascii', 'ignore').decode('ascii')
    folder_name = re.sub(r'[^-\w.]', '_', folder_name)
    return folder_name

def ensure_https(url):
    """Ensure the URL uses HTTPS if supported by the server."""
    parsed_url = urlparse(url)
    if parsed_url.scheme == 'http':
        https_url = urlunparse(parsed_url._replace(scheme='https'))
        try:
            response = requests.head(https_url, timeout=5)
            if response.status_code == 200:
                return https_url
        except requests.RequestException:
            pass
    return url

def download_images(url, folder, visited_urls=None):
    """Download images from the provided URL using requests and BeautifulSoup, ignoring robots.txt and searching recursively."""
    if visited_urls is None:
        visited_urls = set()

    # Avoid revisiting the same URL
    if url in visited_urls:
        return
    visited_urls.add(url)

    try:
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(Fore.RED + f"[ERROR] Failed to retrieve URL: {url}")
            return

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the base domain of the starting URL
        base_domain = urlparse(url).netloc

        # Download images from the current page
        img_tags = soup.find_all('img')
        img_urls = [urljoin(url, img['src']) for img in img_tags if 'src' in img.attrs and img['src'].lower().endswith(ALLOWED_IMAGE_EXTENSIONS)]

        if not img_urls:
            logger.warning(Fore.YELLOW + f"[WARNING] No images found at URL: {url}")
        else:
            if not os.path.exists(folder):
                os.makedirs(folder)

            for img_url in img_urls:
                img_url = ensure_https(img_url)
                try:
                    img_response = requests.get(img_url, stream=True)
                    if img_response.status_code == 200:
                        img_name = safe_filename(os.path.basename(img_url))
                        img_path = os.path.join(folder, img_name)
                        with open(img_path, 'wb') as f:
                            for chunk in img_response.iter_content(1024):
                                f.write(chunk)
                        # Check image size
                        image = cv2.imread(img_path)
                        if image is None or image.shape[0] < MIN_IMAGE_SIZE[1] or image.shape[1] < MIN_IMAGE_SIZE[0]:
                            os.remove(img_path)
                            logger.info(Fore.LIGHTBLACK_EX + f"{img_url} was deleted because it is smaller than {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]} pixels.")
                        else:
                            logger.info(Fore.GREEN + f"[SCRAPING] {img_url}")
                    else:
                        logger.warning(Fore.YELLOW + f"[WARNING] Failed to download image: {img_url}")
                except Exception as e:
                    logger.error(Fore.RED + f"[ERROR] Downloading image: {img_url} - {str(e)}")

        # Find all links on the page and recursively download images
        for link in soup.find_all('a', href=True):
            next_url = urljoin(url, link['href'])
            next_domain = urlparse(next_url).netloc

            # Only follow links within the same domain
            if next_domain == base_domain and next_url not in visited_urls:
                download_images(next_url, folder, visited_urls)

    except Exception as e:
        logger.error(Fore.RED + f"[ERROR] Downloading images from {url}: {str(e)}")

def process_downloaded_images(folder, csv_writer, download_counter):
    """Process downloaded images and create the CSV file."""
    folder = Path(folder)
    if not folder.exists():
        logger.error(Fore.RED + f"[ERROR] The specified folder does not exist: {folder}")
        return

    files = [f for f in folder.rglob('*') if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS]

    if not files:
        logger.warning(Fore.YELLOW + f"[WARNING] No files found in the specified folder: {folder}")
        return

    # List to store files to be deleted
    files_to_delete = []

    def process_image(filepath):
        try:
            num_faces = detect_faces(str(filepath))
            if num_faces > 0:
                creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sha256_hash = sha256_checksum(filepath)
                file_size = filepath.stat().st_size
                logger.info(Fore.CYAN + f"[PROCESSING]: {filepath}")
                with file_lock:
                    csv_writer.writerow([filepath.name, str(filepath), file_size, num_faces, crawl_date, creation_date, sha256_hash])
                    download_counter[0] += 1
                logger.info(Fore.GREEN + f"{filepath.name} contains {num_faces} face(s).")
            else:
                with file_lock:
                    files_to_delete.append(filepath)
                logger.info(Fore.LIGHTBLACK_EX + f"{filepath.name} will be deleted as no faces were detected.")
        except Exception as e:
            logger.error(Fore.RED + f"Error processing image {filepath}: {str(e)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_image, file) for file in files]
        for future in as_completed(futures):
            future.result()  # Wait for all tasks to complete

    # Delete files after processing
    for filepath in files_to_delete:
        try:
            filepath.unlink()
            logger.info(Fore.LIGHTBLACK_EX + f"{filepath.name} was deleted.")
        except Exception as e:
            logger.error(Fore.RED + f"[ERROR] Deleting file {filepath}: {str(e)}")

def remove_duplicates(folder):
    """Remove duplicate images, keeping the one with the highest resolution."""
    files = [f for f in folder.rglob('*') if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS]
    processed_files = []
    files_to_delete = []

    def process_file(file):
        try:
            image = cv2.imread(str(file))
            if image is None:
                return
            for existing_file in processed_files:
                existing_image = cv2.imread(str(existing_file))
                if existing_image is None:
                    continue
                # Resize images to the same size for SSIM comparison
                image_resized = cv2.resize(image, MIN_IMAGE_SIZE)
                existing_image_resized = cv2.resize(existing_image, MIN_IMAGE_SIZE)
                s = ssim(image_resized, existing_image_resized, multichannel=True)
                if s > SSIM_THRESHOLD:  # Threshold for considering images as duplicates
                    if image.shape[0] * image.shape[1] > existing_image.shape[0] * existing_image.shape[1]:
                        files_to_delete.append(existing_file)
                        processed_files.remove(existing_file)
                        processed_files.append(file)
                    else:
                        files_to_delete.append(file)
                    return
            processed_files.append(file)
        except Exception as e:
            logger.error(Fore.RED + f"[ERROR] Processing file {file}: {str(e)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_file, file) for file in files]
        for future in as_completed(futures):
            future.result()  # Wait for all tasks to complete

    # Delete duplicate files after processing
    for filepath in files_to_delete:
        try:
            filepath.unlink()
            logger.info(Fore.LIGHTBLACK_EX + f"{filepath.name} was deleted as a duplicate.")
        except Exception as e:
            logger.error(Fore.RED + f"[ERROR] Deleting file {filepath}: {str(e)}")

if __name__ == "__main__":
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="Crawl a website and download images with faces using requests and BeautifulSoup.")
    parser.add_argument("url", help="The start URL for crawling")
    parser.add_argument("--threads", type=int, default=MAX_WORKERS, help="Number of threads to use for processing images")
    args = parser.parse_args()

    start_url = args.url
    base_folder_name = create_single_folder_name(start_url)
    download_folder = Path(base_folder_name)

    # Create download directory if it doesn't exist
    if not download_folder.exists():
        download_folder.mkdir()

    csv_file = download_folder / f"{CSV_FILENAME_PREFIX}{base_folder_name}.csv"

    # Download images using requests and BeautifulSoup
    download_images(start_url, str(download_folder))

    # Process downloaded images and create CSV
    download_counter = [0]
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Filename", "Source_URL", "File_Size", "Number_of_Faces", "Crawl_Date", "Creation_Date", "SHA256"])
        process_downloaded_images(download_folder, csv_writer, download_counter)

    # Remove duplicate images
    remove_duplicates(download_folder)

    elapsed_time = time.perf_counter() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logger.info(Fore.CYAN + f"Total images downloaded and saved: {download_counter[0]}")
    logger.info(Fore.CYAN + f"Total runtime: {int(minutes)} minutes and {int(seconds)} seconds")