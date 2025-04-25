import os
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageFilter
import cv2
from urllib.parse import urlparse, urljoin, urlunparse
import argparse
import logging
from colorama import Fore, init
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
MIN_IMAGE_SIZE = (300, 300)  # Minimum image size (width, height)
FACE_PADDING_RATIO = 0.5  # Padding around the face (50% of face size)
BLUR_RADIUS = 10  # Gaussian blur radius for background
SSIM_THRESHOLD = 0.95  # Threshold for considering images as duplicates
MAX_WORKERS = 4  # Number of threads for parallel processing

# Gesichtserkennung mit OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

def safe_filename(filename):
    """Create a safe filename by removing invalid characters."""
    keep_chars = (' ', '.', '_', '-')
    return "".join(c for c in filename if c.isalnum() or c in keep_chars).rstrip()

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
                            detect_faces(img_path, folder)
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

def detect_faces(image_path, folder):
    """Detect faces in an image, crop them with padding, blur the background, and save the results."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Vergrößere den Ausschnitt um das Gesicht herum
        padding = int(max(w, h) * FACE_PADDING_RATIO)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        # Schneide das Gesicht mit Rand aus
        face_with_padding = image[y1:y2, x1:x2]

        # Pixelung des Hintergrunds
        pil_image = Image.fromarray(cv2.cvtColor(face_with_padding, cv2.COLOR_BGR2RGB))
        blurred_background = pil_image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        blurred_background.paste(pil_image, (0, 0), mask=None)  # Gesicht über den gepixelten Hintergrund legen

        # Speichere das Ergebnis
        face_image_path = os.path.join(folder, f"face_{x}_{y}.jpg")
        blurred_background.save(face_image_path)
        logger.info(Fore.GREEN + f"[FACE] Saved face to {face_image_path}")

def remove_duplicates(folder):
    """Remove duplicate images, keeping the one with the highest resolution."""
    folder = Path(folder)
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
    parser = argparse.ArgumentParser(description='Download images from a website recursively.')
    parser.add_argument('url', type=str, help='The URL of the website to download images from.')
    args = parser.parse_args()

    # Create folder name based on the domain
    folder_name = urlparse(args.url).netloc.replace('www.', '').replace('.', '_')
    download_images(args.url, folder_name)

    # Remove duplicates after downloading
    remove_duplicates(folder_name)