import os
import time
from typing import List
import mss
import numpy as np
from PIL import Image

from openrecall.config import screenshots_path, args
from openrecall.database import insert_entry
from openrecall.nlp import get_embedding
from openrecall.ocr import extract_text_from_image
from openrecall.utils import (
    get_active_app_name,
    get_active_window_title,
    is_user_active,
)

# make sure folder exists
os.makedirs(screenshots_path, exist_ok=True)

def take_screenshots() -> List[np.ndarray]:
    screenshots = []
    with mss.mss() as sct:
        monitor_indices = [1] if args.primary_monitor_only else range(1, len(sct.monitors))
        for i in monitor_indices:
            if i < len(sct.monitors):
                sct_img = sct.grab(sct.monitors[i])
                screenshot = np.array(sct_img)[:, :, [2, 1, 0]]  # BGRA -> RGB
                screenshots.append(screenshot)
    return screenshots

def mean_structured_similarity_index(img1: np.ndarray, img2: np.ndarray, L: int = 255) -> float:
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1*L)**2, (K2*L)**2
    def rgb2gray(img): return 0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]
    img1_gray, img2_gray = rgb2gray(img1), rgb2gray(img2)
    mu1, mu2 = np.mean(img1_gray), np.mean(img2_gray)
    sigma1_sq, sigma2_sq = np.var(img1_gray), np.var(img2_gray)
    sigma12 = np.mean((img1_gray - mu1)*(img2_gray - mu2))
    return ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2+mu2**2+C1)*(sigma1_sq+sigma2_sq+C2))

def is_similar(img1: np.ndarray, img2: np.ndarray, threshold: float = 0.9) -> bool:
    return mean_structured_similarity_index(img1, img2) >= threshold

def record_screenshots_thread():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    last_screenshots = take_screenshots()

    while True:
        if not is_user_active():
            time.sleep(3)
            continue

        current_screenshots = take_screenshots()

        # handle monitor count changes
        if len(current_screenshots) != len(last_screenshots):
            last_screenshots = current_screenshots
            time.sleep(3)
            continue

        for i, screenshot in enumerate(current_screenshots):
            last_screenshot = last_screenshots[i]

            if not is_similar(screenshot, last_screenshot):
                last_screenshots[i] = screenshot

                timestamp = int(time.time())
                filename = f"{timestamp}.webp"
                filepath = os.path.join(screenshots_path, filename)

                # save image
                Image.fromarray(screenshot).save(filepath, format="webp", lossless=True)
                print(f"Saved screenshot: {filepath}")

                # extract text and insert into DB
                text = extract_text_from_image(screenshot)
                if text.strip():
                    embedding = get_embedding(text)
                    active_app_name = get_active_app_name() or "Unknown App"
                    active_window_title = get_active_window_title() or "Unknown Title"
                    insert_entry(
                        text, timestamp, embedding, active_app_name, active_window_title, filename
                    )

        time.sleep(3)
