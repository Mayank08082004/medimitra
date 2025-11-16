"""
generate_plots.py

Evaluation and plotting script for the Medimitra research paper.

This script operates in two modes:
1. 'evaluate': Runs API against a ground truth dataset and saves metrics.
2. 'plot':     Loads saved metrics and generates all paper-ready plots.

Author: Your Name
Date: 2025-11-15
"""

import os
import json
import re
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# --- CONFIGURATION ---

# Directory to save generated plots
PLOTS_DIR = Path("figures")

# API endpoint for the Medimitra server
API_URL = "http://127.0.0.1:5001/process"

# Fields to evaluate (keys in your 'ground_truth' JSONs)
# 'medications' is handled separately.
FIELDS_TO_EVALUATE = [
    "patient_name",
    "patient_dob",
    "prescriber_name",
    "prescriber_dea",
    "date_written",
    "diagnosis",
]

# --- HELPER FUNCTIONS ---

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation/spaces for comparison."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def get_api_prediction(image_path: Path) -> Tuple[Dict[str, Any], float]:
    """
    Calls the /process endpoint and returns the extraction and processing time.
    """
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")} # Adjust mime if needed
            start_time = time.perf_counter()
            response = requests.post(API_URL, files=files)
            end_time = time.perf_counter()
            
            processing_time_ms = (end_time - start_time) * 1000

            if response.status_code == 200:
                return response.json().get("extraction", {}), processing_time_ms
            else:
                print(f"  [!] API Error {response.status_code} for {image_path.name}: {response.text}")
                return {}, processing_time_ms
    except Exception as e:
        print(f"  [!] Request failed for {image_path.name}: {e}")
        return {}, 0.0

def safe_get_nested(data: Dict, path: List[str], default: Any = None) -> Any:
    """
    Safely get a nested value from a dict using a list of keys.
    e.g., safe_get_nested(data, ["patient", "name"])
    """
    try:
        for key in path:
            data = data[key]
        return data
    except (KeyError, TypeError):
        return default

# --- 1. EVALUATION MODE ---

def run_evaluation(eval_images_dir: Path, ground_truth_dir: Path) -> List[Dict]:
    """
    Runs evaluation against the test dataset.
    Compares API output to ground truth.
    """
    print(f"Starting evaluation...")
    print(f"  API URL: {API_URL}")
    print(f"  Images dir: {eval_images_dir}")
    print(f"  Ground Truth dir: {ground_truth_dir}")

    gt_files = list(ground_truth_dir.glob("*.json"))
    if not gt_files:
        print(f"Error: No ground truth .json files found in {ground_truth_dir}")
        return []

    all_results = []
    
    for gt_path in tqdm(gt_files, desc="Evaluating Documents"):
        doc_id = gt_path.stem
        image_path = eval_images_dir / f"{doc_id}.png" # Assumes .png, adjust if needed
        
        if not image_path.exists():
            image_path = eval_images_dir / f"{doc_id}.jpg" # Try .jpg
            if not image_path.exists():
                image_path = eval_images_dir / f"{doc_id}.pdf" # Try .pdf
                if not image_path.exists():
                    print(f"  [!] Skipping {doc_id}: No matching image found.")
                    continue

        # Load ground truth
        with open(gt_path, "r") as f:
            gt_data = json.load(f)

        # Get API prediction
        api_extraction, proc_time = get_api_prediction(image_path)
        
        doc_metrics = {
            "document_id": doc_id,
            "processing_time_ms": proc_time,
            "fields": {},
            "medications": {},
        }

        # --- A. Evaluate simple key-value fields ---
        for field in FIELDS_TO_EVALUATE:
            gt_value = normalize_text(gt_data.get(field))
            
            # Map simple field names to nested API paths
            api_path_map = {
                "patient_name": ["patient", "name"],
                "patient_dob": ["patient", "dob"],
                "prescriber_name": ["prescriber", "name"],
                "prescriber_dea": ["prescriber", "dea"],
                "date_written": ["metadata", "date_written"],
                "diagnosis": ["clinical_info", "diagnosis"], # Assumes diagnosis is a single string
            }
            
            # Handle list-based diagnosis (take first)
            if field == "diagnosis":
                api_diag = safe_get_nested(api_extraction, api_path_map[field], [])
                api_value = normalize_text(api_diag[0] if api_diag else None)
            else:
                api_value = normalize_text(safe_get_nested(api_extraction, api_path_map[field]))

            is_match = 1 if (api_value == gt_value) else 0
            
            # For 'diagnosis', count a match if GT is empty and API is empty
            if field == "diagnosis" and not gt_value and not api_value:
                is_match = 1
                
            doc_metrics["fields"][field] = {
                "true": gt_value,
                "pred": api_value,
                "match": is_match,
            }

        # --- B. Evaluate medication list ---
        gt_meds = gt_data.get("medications", [])
        api_meds = api_extraction.get("medications", [])

        gt_med_names = set(normalize_text(med.get("name")) for med in gt_meds if med.get("name"))
        api_med_names = set(normalize_text(med.get("name")) for med in api_meds if med.get("name"))
        
        tp = len(gt_med_names.intersection(api_med_names))
        fp = len(api_med_names - gt_med_names)
        fn = len(gt_med_names - api_med_names)
        
        doc_metrics["medications"] = {"tp": tp, "fp": fp, "fn": fn}
        all_results.append(doc_metrics)

    print(f"\nEvaluation complete. Processed {len(all_results)} documents.")
    return all_results

# --- 2. PLOTTING MODE ---

def calculate_aggregate_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates P/R/F1 from the flattened results."""
    all_metrics = []

    # 1. Field metrics
    for field in FIELDS_TO_EVALUATE:
        matches = results_df[f"field_{field}_match"]
        # TP = sum(matches == 1)
        # Total = len(matches)
        # We'll treat this as a simple accuracy:
        accuracy = matches.sum() / len(matches)
        
        # A more robust P/R/F1 (for presence/absence)
        y_true = [1 if bool(t) else 0 for t in results_df[f"field_{field}_true"]]
        y_pred = [1 if bool(p) else 0 for p in results_df[f"field_{field}_pred"]]
        
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        all_metrics.append({
            "entity": field.replace("_", " ").title(),
            "Precision": p,
            "Recall": r,
            "F1-Score": f1,
            "Exact Match Accuracy": accuracy
        })

    # 2. Medication metrics
    med_tps = results_df["med_tp"].sum()
    med_fps = results_df["med_fp"].sum()
    med_fns = results_df["med_fn"].sum()

    med_p = med_tps / (med_tps + med_fps) if (med_tps + med_fps) > 0 else 0
    med_r = med_tps / (med_tps + med_fns) if (med_tps + med_fns) > 0 else 0
    med_f1 = (2 * med_p * med_r) / (med_p + med_r) if (med_p + med_r) > 0 else 0

    all_metrics.append({
        "entity": "Medication Names",
        "Precision": med_p,
        "Recall": med_r,
        "F1-Score": med_f1,
        "Exact Match Accuracy": None # N/A for list
    })
    
    return pd.DataFrame(all_metrics).set_index("entity")

def flatten_results(results: List[Dict]) -> pd.DataFrame:
    """Flattens the complex result list into a DataFrame."""
    flattened = []
    for doc in results:
        row = {
            "document_id": doc["document_id"],
            "processing_time_ms": doc["processing_time_ms"],
            "med_tp": doc["medications"]["tp"],
            "med_fp": doc["medications"]["fp"],
            "med_fn": doc["medications"]["fn"],
        }
        for field, metrics in doc["fields"].items():
            row[f"field_{field}_true"] = metrics["true"]
            row[f"field_{field}_pred"] = metrics["pred"]
            row[f"field_{field}_match"] = metrics["match"]
        flattened.append(row)
    return pd.DataFrame(flattened)

def plot_f1_scores(metrics_df: pd.DataFrame, save_path: Path):
    """Plots the main F1-Score grouped bar chart."""
    print("  Generating F1-Score plot...")
    plot_df = metrics_df[["Precision", "Recall", "F1-Score"]]
    
    plt.figure(figsize=(14, 8))
    plot_df.plot(
        kind="bar",
        rot=45,
        width=0.8,
        colormap="viridis",
        edgecolor="black"
    )
    plt.title("Extraction Performance by Entity (F1-Score)", fontsize=18, pad=20)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("Entity", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Metric", loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_latency_distribution(results_df: pd.DataFrame, save_path: Path):
    """Plots a histogram of API processing times."""
    print("  Generating Latency plot...")
    times_ms = results_df["processing_time_ms"]
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
    plt.savefig(save_path)
    plt.close()

def plot_llm_comparison(regex_metrics: pd.DataFrame, llm_metrics: pd.DataFrame, save_path: Path):
    """Plots the 'Regex Only' vs 'Regex + LLM' comparison."""
    print("  Generating LLM Comparison plot...")
    
    # Get F1 scores for both
    regex_f1 = regex_metrics["F1-Score"].rename("Regex Only")
    llm_f1 = llm_metrics["F1-Score"].rename("Regex + LLM")
    
    compare_df = pd.concat([regex_f1, llm_f1], axis=1)
    
    plt.figure(figsize=(14, 8))
    compare_df.plot(
        kind="bar",
        rot=45,
        width=0.8,
        colormap="Paired",
        edgecolor="black"
    )
    plt.title("Value of LLM Refinement (F1-Score)", fontsize=18, pad=20)
    plt.ylabel("F1-Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("Entity", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Processing Mode")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_qualitative_table(results_df: pd.DataFrame, save_path: Path):
    """Saves a Markdown table of the worst-performing examples."""
    print("  Generating Qualitative Error table...")
    
    error_rows = []
    for field in FIELDS_TO_EVALUATE:
        field_errors = results_df[results_df[f"field_{field}_match"] == 0]
        for _, row in field_errors.iterrows():
            error_rows.append({
                "Document ID": row["document_id"],
                "Field": field.replace("_", " ").title(),
                "Ground Truth": row[f"field_{field}_true"],
                "Predicted": row[f"field_{field}_pred"],
            })
            
    if not error_rows:
        print("  No errors found. Skipping table.")
        return

    error_df = pd.DataFrame(error_rows).head(20) # Limit to 20 examples
    
    with open(save_path, "w") as f:
        f.write("# Qualitative Error Analysis\n\n")
        f.write("A sample of fields where the predicted value did not match the ground truth.\n\n")
        error_df.to_markdown(f, index=False)
        
    print(f"  Saved error table to {save_path}")

def run_plotting(
    llm_results_path: Path,
    regex_results_path: Optional[Path] = None
):
    """
    Main plotting function. Loads results file(s) and generates all plots.
    """
    print(f"Loading results from: {llm_results_path}")
    if not llm_results_path.exists():
        print(f"Error: Results file not found: {llm_results_path}")
        return

    with open(llm_results_path, "r") as f:
        llm_results = json.load(f)

    llm_df = flatten_results(llm_results)
    llm_metrics_df = calculate_aggregate_metrics(llm_df)
    
    print("\n--- LLM + Regex Metrics ---")
    print(llm_metrics_df)
    
    # --- Generate single-run plots ---
    PLOTS_DIR.mkdir(exist_ok=True)
    
    plot_f1_scores(
        llm_metrics_df,
        PLOTS_DIR / "plot_f1_scores.png"
    )
    
    plot_latency_distribution(
        llm_df,
        PLOTS_DIR / "plot_latency_distribution.png"
    )
    
    save_qualitative_table(
        llm_df,
        PLOTS_DIR / "table_qualitative_errors.md"
    )
    
    # --- Generate comparison plot if 'regex_only' file is provided ---
    if regex_results_path:
        if not regex_results_path.exists():
            print(f"Warning: Regex-only results file not found: {regex_results_path}. Skipping comparison plot.")
        else:
            print(f"\nLoading regex-only results from: {regex_results_path}")
            with open(regex_results_path, "r") as f:
                regex_results = json.load(f)
            
            regex_df = flatten_results(regex_results)
            regex_metrics_df = calculate_aggregate_metrics(regex_df)
            
            print("\n--- Regex-Only Metrics ---")
            print(regex_metrics_df)
            
            plot_llm_comparison(
                regex_metrics=regex_metrics_df,
                llm_metrics=llm_metrics_df,
                save_path=PLOTS_DIR / "plot_llm_comparison.png"
            )

    print(f"\nAll plots saved to {PLOTS_DIR}/")

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Medimitra Evaluation & Plotting Script")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- 'evaluate' sub-parser ---
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation against the API.")
    eval_parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing evaluation images (e.g., 'sample_data/')."
    )
    eval_parser.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="Directory containing ground truth JSON files."
    )
    eval_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the JSON results file (e.g., 'results_llm.json')."
    )

    # --- 'plot' sub-parser ---
    plot_parser = subparsers.add_parser("plot", help="Generate plots from saved results.")
    plot_parser.add_argument(
        "--results_llm",
        type=Path,
        required=True,
        help="Path to the main (Regex + LLM) JSON results file."
    )
    plot_parser.add_argument(
        "--results_regex",
        type=Path,
        default=None,
        help="Optional. Path to the 'Regex Only' JSON results file for comparison."
    )

    args = parser.parse_args()

    # --- Execute selected mode ---
    if args.mode == "evaluate":
        results = run_evaluation(args.images, args.truth)
        if results:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Successfully saved evaluation results to {args.output}")
        else:
            print("Evaluation failed or produced no results.")

    elif args.mode == "plot":
        run_plotting(args.results_llm, args.results_regex)

if __name__ == "__main__":
    main()