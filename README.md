# faceget# 🌐 Web Image Scraper with Face Detection  😊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A powerful Python-based web scraping tool that downloads images from a website, detects faces using OpenCV, and organizes the results into a CSV file. Perfect for data collection, research, or personal projects! 🚀


## ✨ Features

- **🌐 Web Scraping**: Downloads images (recursive) from a specified URL.
- **👤 Face Detection**: Uses OpenCV's Haar Cascade to detect faces in images.
- **🧹 Duplicate Removal**: Removes duplicate images based on structural similarity (SSIM).
- **📊 CSV Export**: Saves metadata (filename, source URL, file size, number of faces, etc.) in a CSV file.
- **🔍 Recursive Crawling**: Optionally crawls linked pages to find more images.
- **⚡ Threaded Processing**: Utilizes multi-threading for faster image processing.
- **📂 Organized Output**: Saves images and metadata in a structured folder.
- ⚡ **Multithreading**: Utilizes multiple threads for parallel image processing for better performance


## 🛠️ Installation

### Prerequisites

- **Python 3.8 or higher** 🐍
- **pip** (Python package manager)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gill-Bates/faceget.git
   cd faceget
2. **Make sure you have Python 3.x installed and use `pip` to install the dependencies:**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage 🚀

```bash
python faceget.py <URL> [--threads <number_of_threads>]
```

- `<URL>`: The starting URL to crawl
- `--threads`: Optional. Number of threads to use. Defaults to the number of CPU cores.

### Examples

Crawl a website using the default number of threads:

```bash
python faceget.py https://example.com
```

Crawl a website using 8 threads:

```bash
python faceget.py https://example.com --threads 8
```

## License 📄

This project is licensed under the [MIT License](LICENSE).

---

### Disclaimer 🔍

Use this tool at your own risk. The author is not responsible for any direct or indirect damages that may occur due to the use of this tool.

---


Feel free to open an issue or submit a pull request to contribute to this project. 🚀✨
