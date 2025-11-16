import os
import requests
from pathlib import Path
from tqdm import tqdm
import json

# --- CONFIG ---
IMAGE_DIR = Path("evaluation_images")
OUTPUT_DIR = Path("ground_truth")
API_URL = "http://127.0.0.1:5001/process"
# ----------------

def process_images():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    image_files = list(IMAGE_DIR.glob("*.png")) + \
                  list(IMAGE_DIR.glob("*.jpg")) + \
                  list(IMAGE_DIR.glob("*.pdf"))

    print(f"Found {len(image_files)} images to process...")

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            with open(img_path, "rb") as f:
                files = {"file": (img_path.name, f)}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    # Save the *full* API response
                    data = response.json()
                    output_filename = OUTPUT_DIR / f"{img_path.stem}_api_output.json"
                    with open(output_filename, "w") as out_f:
                        json.dump(data, out_f, indent=2)
                else:
                    print(f"Error on {img_path.name}: {response.status_code}")
        
        except Exception as e:
            print(f"Failed {img_path.name}: {e}")

if __name__ == "__main__":
    # Make sure your 'app.py' server is running first!
    process_images()