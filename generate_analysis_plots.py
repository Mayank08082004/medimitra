"""
generate_analysis_plots.py

Runs performance and behavior analysis for the Medimitra paper.
v2.0: Now supports resume/batching to prevent overheating.

This script does NOT require a ground truth dataset.
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from tqdm import tqdm

# --- CONFIGURATION ---
PLOTS_DIR = Path("figures")
API_URL = "http://127.0.0.1:5001/process"

# Fields to check for "found rate"
FIELDS_TO_ANALYZE = {
    "Patient Name": ["extraction", "patient", "name"],
    "Patient DOB": ["extraction", "patient", "dob"],
    "Prescriber Name": ["extraction", "prescriber", "name"],
    "Prescriber DEA": ["extraction", "prescriber", "dea"],
    "Date Written": ["extraction", "metadata", "date_written"],
    "Diagnosis": ["extraction", "clinical_info", "diagnosis"],
    "Medications": ["extraction", "medications"],
}

# --- HELPER FUNCTIONS ---

def safe_get_nested(data: Dict, path: List[str]) -> Any:
    """Safely get a nested value from a dict."""
    try:
        for key in path:
            data = data[key]
        return data
    except (KeyError, TypeError, IndexError):
        return None

# --- SCRIPT MODES ---

def run_api_analysis(images_dir: Path, output_file: Path):
    """
    Calls the /process endpoint for all images.
    Can be safely stopped and resumed.
    """
    print(f"Starting API analysis on: {images_dir}")
    print(f"Saving/Appending results to: {output_file}")
    
    # --- BATCH PROCESSING FIX ---
    all_responses = []
    processed_doc_names = set()
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                all_responses = json.load(f)
            processed_doc_names = set(item.get("_meta_doc_name") for item in all_responses)
            print(f"Loaded {len(all_responses)} existing results. Will skip these.")
        except json.JSONDecodeError:
            print(f"Warning: Could not read {output_file}. Starting from scratch.")
            all_responses = []
            processed_doc_names = set()
    # --- END FIX ---
    
    image_files = list(images_dir.glob("*.png")) + \
                  list(images_dir.glob("*.jpg")) + \
                  list(images_dir.glob("*.jpeg")) + \
                  list(images_dir.glob("*.tiff")) + \
                  list(images_dir.glob("*.pdf"))

    if not image_files:
        print(f"Error: No images found in {images_dir}")
        return

    # Filter out images that are already processed
    images_to_process = [
        img for img in image_files 
        if img.name not in processed_doc_names
    ]
    
    if not images_to_process:
        print("All images have already been processed.")
        return

    print(f"Total images: {len(image_files)}. Already processed: {len(processed_doc_names)}. Remaining: {len(images_to_process)}")

    # Use a new list for the *current* run's output, to be appended
    new_responses_this_run = []
    try:
        for img_path in tqdm(images_to_process, desc="Analyzing Documents"):
            try:
                with open(img_path, "rb") as f:
                    mime_type = "image/jpeg"
                    if img_path.suffix == ".png": mime_type = "image/png"
                    elif img_path.suffix == ".pdf": mime_type = "application/pdf"
                    elif img_path.suffix in [".tiff", ".tif"]: mime_type = "image/tiff"
                            
                    files = {"file": (img_path.name, f, mime_type)}
                    
                    start_time = time.perf_counter()
                    response = requests.post(API_URL, files=files)
                    end_time = time.perf_counter()
                    
                    processing_time_ms = (end_time - start_time) * 1000

                    if response.status_code == 200:
                        api_data = response.json()
                        api_data["_meta_processing_time_ms"] = processing_time_ms
                        api_data["_meta_doc_name"] = img_path.name
                        new_responses_this_run.append(api_data)
                    else:
                        print(f"  [!] Error on {img_path.name}: {response.status_code}")

            except Exception as e:
                print(f"  [!] Failed {img_path.name}: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")
    
    finally:
        # Save all results (old + new)
        all_responses.extend(new_responses_this_run)
        with open(output_file, "w") as f:
            json.dump(all_responses, f, indent=2)
        print(f"\nAnalysis complete (or paused). Saved {len(all_responses)} total results to {output_file}")


def run_plotting(results_file: Path):
    """
    Loads the saved API responses and generates all plots.
    """
    print(f"Loading results from {results_file}...")
    PLOTS_DIR.mkdir(exist_ok=True)

    try:
        with open(results_file, "r") as f:
            results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Error: Could not load or find {results_file}.")
        return

    if not results:
        print("Error: Results file is empty. No plots generated.")
        return

    df = pd.DataFrame(results)

    # --- 1. Plot Latency ---
    print("  Generating Latency plot...")
    try:
        times_ms = df["_meta_processing_time_ms"].dropna()
        times_s = times_ms / 1000.0
        
        avg_time = times_s.mean()
        p95_time = times_s.quantile(0.95)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(times_s, bins=30, kde=True, color="teal")
        plt.axvline(avg_time, color='red', linestyle='--', label=f"Average: {avg_time:.2f}s")
        plt.axvline(p95_time, color='orange', linestyle='--', label=f"95th Pctl: {p95_time:.2f}s")
        plt.title("Distribution of API Processing Latency", fontsize=16, pad=15)
        plt.xlabel("Processing Time (seconds)", fontsize=12)
        plt.ylabel("Frequency (Count of Documents)", fontsize=12)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "plot_latency_distribution.png")
        plt.close()
    except Exception as e:
        print(f"  [!] Failed to generate latency plot: {e}")

    # --- 2. Plot OCR Confidence ---
    print("  Generating OCR Confidence plot...")
    try:
        conf_scores = df["ocr_confidence"].dropna()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(conf_scores, bins=25, kde=True, color="indigo")
        plt.title("Distribution of OCR Confidence Scores", fontsize=16, pad=15)
        plt.xlabel("Confidence Score (0.0 to 1.0)", fontsize=12)
        plt.ylabel("Frequency (Count of Documents)", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "plot_ocr_confidence.png")
        plt.close()
    except Exception as e:
        print(f"  [!] Failed to generate confidence plot: {e}")

    # --- 3. Plot Extraction Rate ---
    print("  Generating Extraction Rate plot...")
    try:
        extraction_rates = []
        total_docs = len(df)
        
        for field_name, json_path in FIELDS_TO_ANALYZE.items():
            found_count = df.apply(lambda row: bool(safe_get_nested(row, json_path)), axis=1).sum()
            rate = (found_count / total_docs) * 100
            extraction_rates.append({"Field": field_name, "Found Rate (%)": rate})

        rate_df = pd.DataFrame(extraction_rates)
        
        plt.figure(figsize=(12, 7))
        sns.barplot(x="Found Rate (%)", y="Field", data=rate_df, palette="viridis")
        plt.title("Field Extraction Rate (Throughput)", fontsize=16, pad=15)
        plt.xlabel("Documents (%)", fontsize=12)
        plt.ylabel("Field", fontsize=12)
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "plot_extraction_rate.png")
        plt.close()
    except Exception as e:
        print(f"  [!] Failed to generate extraction rate plot: {e}")
        
    print(f"\nAll plots saved to {PLOTS_DIR}/")

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Medimitra Performance Analysis Script (No-Truth)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- 'analyze' sub-parser ---
    eval_parser = subparsers.add_parser("analyze", help="Run API on images and save all JSON responses.")
    eval_parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing images (e.g., 'analysis_data/images/')."
    )
    eval_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the JSON results file (e.g., 'api_responses.json')."
    )

    # --- 'plot' sub-parser ---
    plot_parser = subparsers.add_parser("plot", help="Generate plots from a saved results file.")
    plot_parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to the JSON results file (e.g., 'api_responses.json')."
    )

    args = parser.parse_args()

    if args.mode == "analyze":
        run_api_analysis(args.images, args.output)
    elif args.mode == "plot":
        run_plotting(args.results)

if __name__ == "__main__":
    main()