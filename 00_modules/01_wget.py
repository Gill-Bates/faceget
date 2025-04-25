import os
import subprocess
import concurrent.futures
from urllib.parse import urlparse
from PIL import Image

def create_output_dir(url):
    """Erstellt ein Verzeichnis basierend auf dem Domain-Namen."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc  # Extrahiert die Domain (z.B. "example.com")
    output_dir = domain.replace("www.", "")  # Entfernt "www." falls vorhanden
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def download_images_with_wget(url, output_dir):
    """Lädt Bilder rekursiv mit wget herunter, begrenzt auf interne Links und no-parent."""
    command = [
        "wget",
        "-r",  # Rekursiver Download
        "-l", "inf",  # Unbegrenzte Rekursionstiefe
        "-nd",  # Keine Verzeichnisstruktur erstellen
        "-A", "jpg,jpeg,png,gif,webp",  # Nur Bilddateien herunterladen
        "-P", output_dir,  # Zielverzeichnis für die Downloads
        "-e", "robots=off",  # Ignoriert robots.txt
        "--no-parent",  # Verhindert das Crawlen von übergeordneten Verzeichnissen
        "--domains", urlparse(url).netloc,  # Beschränkt das Crawlen auf die angegebene Domain
        url
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Download abgeschlossen. Bilder wurden in '{output_dir}' gespeichert.")
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Herunterladen der Bilder: {e}")

def filter_images_by_size(output_dir, min_width=300, min_height=300):
    """Löscht Bilder, die kleiner als die angegebene Mindestgröße sind."""
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                if width < min_width or height < min_height:
                    os.remove(file_path)
                    print(f"Gelöscht: {filename} (zu klein: {width}x{height})")
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {filename}: {e}")

def download_and_filter_images(url):
    """Hauptfunktion: Lädt Bilder herunter und filtert sie nach Größe."""
    # Erstelle das Ausgabeverzeichnis
    output_dir = create_output_dir(url)

    # Lade Bilder mit wget herunter
    download_images_with_wget(url, output_dir)

    # Filtere Bilder nach Größe
    filter_images_by_size(output_dir)

def main():
    import sys
    if len(sys.argv) != 2:
        print("Verwendung: python script.py <URL>")
        return

    url = sys.argv[1]
    if not url.startswith(("http://", "https://")):
        url = "https://" + url  # Füge automatisch HTTPS hinzu, falls nicht vorhanden

    # Parallelisiere den Download und die Filterung
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(download_and_filter_images, url)

if __name__ == "__main__":
    main()