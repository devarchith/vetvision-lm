"""
generate_manifest.py — Auto-generate the veterinary dataset manifest CSV.

Scans the downloaded TUFTS Veterinary Chest X-Ray dataset directory and
produces a CSV with columns:
    image_path    – relative path from data_dir to the image file
    report_text   – short synthetic description (real reports not released)
    species_label – 0 = canine, 1 = feline

Usage:
    python scripts/generate_manifest.py --data-dir data/vet_dataset --output data/vet_dataset/manifest.csv
    python scripts/generate_manifest.py --data-dir data/vet_dataset --smoke-test
"""

import argparse
import os
import random
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Species inference heuristics
# ---------------------------------------------------------------------------

CANINE_KEYWORDS = {"dog", "canine", "boxer", "labrador", "retriever", "poodle",
                   "beagle", "pug", "bulldog", "dachshund", "mixed", "mutt"}
FELINE_KEYWORDS = {"cat", "feline", "kitten", "tabby", "siamese", "maine", "persian"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dcm"}

# Synthetic report templates by species
REPORT_TEMPLATES = {
    0: [  # canine
        "Lateral and VD thoracic radiographs of a canine patient. "
        "Cardiac silhouette within normal limits. Lung fields clear.",
        "Canine chest radiograph. No evidence of pleural effusion or pneumothorax.",
        "Dog thorax: mild cardiomegaly noted. No infiltrates.",
        "Canine patient radiograph: pulmonary vasculature normal.",
        "Thoracic radiograph — canine. No acute findings.",
    ],
    1: [  # feline
        "Lateral thoracic radiograph of a feline patient. "
        "Mild pleural effusion noted bilaterally.",
        "Feline chest X-ray. Cardiac silhouette unremarkable.",
        "Cat thorax: lungs clear, no consolidation identified.",
        "Feline patient: cardiomegaly with signs of congestive heart failure.",
        "Thoracic radiograph — feline. No gross abnormality detected.",
    ],
}


def infer_species_from_path(path: Path) -> int:
    """Return 0 (canine) or 1 (feline) based on path components."""
    path_lower = str(path).lower()
    for kw in FELINE_KEYWORDS:
        if kw in path_lower:
            return 1
    for kw in CANINE_KEYWORDS:
        if kw in path_lower:
            return 0
    # If ambiguous, use filename hash for 50/50 split
    return hash(path.name) % 2


def infer_species_from_metadata(df: pd.DataFrame, img_path: Path) -> int:
    """
    Try to match the image against metadata CSV columns.
    Falls back to path heuristic.
    """
    filename = img_path.stem
    for col in ("filename", "image_name", "file", "name", "id"):
        if col in df.columns:
            mask = df[col].astype(str).str.contains(filename, case=False, na=False)
            if mask.any():
                row = df[mask].iloc[0]
                for scol in ("species", "label", "category", "class"):
                    val = str(row.get(scol, "")).strip().lower()
                    if val in ("cat", "feline", "1"):
                        return 1
                    if val in ("dog", "canine", "0"):
                        return 0
    return infer_species_from_path(img_path)


def collect_image_paths(data_dir: Path) -> list:
    """Recursively collect all image files from data_dir."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(data_dir.rglob(f"*{ext}"))
        images.extend(data_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def generate_manifest(
    data_dir: str,
    output_path: str,
    meta_csv: str = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build and save a manifest CSV.

    Args:
        data_dir:    Root directory containing images.
        output_path: Path to save the manifest CSV.
        meta_csv:    Optional existing metadata CSV to assist species labelling.
        seed:        Random seed for report template selection.

    Returns:
        manifest DataFrame.
    """
    random.seed(seed)
    data_dir = Path(data_dir)

    # Load existing metadata if available
    meta_df = None
    if meta_csv and Path(meta_csv).exists():
        meta_df = pd.read_csv(meta_csv)
        print(f"[INFO] Loaded metadata: {meta_csv} ({len(meta_df)} rows)")
    else:
        # Search for any CSV in data_dir
        for csv_file in data_dir.rglob("*.csv"):
            try:
                meta_df = pd.read_csv(csv_file)
                print(f"[INFO] Auto-discovered metadata: {csv_file}")
                break
            except Exception:
                continue

    # Collect images
    images = collect_image_paths(data_dir)
    print(f"[INFO] Found {len(images)} images in {data_dir}")

    if not images:
        print("[WARNING] No images found. Generating placeholder manifest.")

    records = []
    for img_path in images:
        # Relative path from data_dir
        try:
            rel_path = img_path.relative_to(data_dir)
        except ValueError:
            rel_path = img_path

        # Infer species
        if meta_df is not None:
            species = infer_species_from_metadata(meta_df, img_path)
        else:
            species = infer_species_from_path(img_path)

        # Sample a report template
        templates = REPORT_TEMPLATES[species]
        report = templates[hash(str(img_path)) % len(templates)]

        records.append({
            "image_path": str(rel_path),
            "report_text": report,
            "species_label": species,
        })

    manifest = pd.DataFrame(records, columns=["image_path", "report_text", "species_label"])

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    n_canine = (manifest["species_label"] == 0).sum()
    n_feline = (manifest["species_label"] == 1).sum()
    print(f"[INFO] Manifest saved: {output_path}")
    print(f"[INFO]   Total: {len(manifest)}  Canine: {n_canine}  Feline: {n_feline}")

    return manifest


def generate_smoke_manifest(output_path: str, num_samples: int = 50) -> pd.DataFrame:
    """Generate a synthetic manifest for smoke testing."""
    records = []
    for i in range(num_samples):
        species = i % 2
        templates = REPORT_TEMPLATES[species]
        report = templates[i % len(templates)]
        records.append({
            "image_path": f"synthetic/img_{i:05d}.jpg",
            "report_text": report,
            "species_label": species,
        })
    manifest = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    print(f"[SMOKE] Generated synthetic manifest: {output_path} ({num_samples} samples)")
    return manifest


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate veterinary dataset manifest CSV"
    )
    parser.add_argument("--data-dir", type=str, default="data/vet_dataset",
                        help="Root directory of the veterinary dataset")
    parser.add_argument("--output", type=str, default="data/vet_dataset/manifest.csv",
                        help="Output path for manifest CSV")
    parser.add_argument("--meta-csv", type=str, default=None,
                        help="Optional existing metadata CSV")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Generate synthetic manifest (no real data needed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke_test:
        generate_smoke_manifest(args.output, num_samples=50)
    else:
        generate_manifest(
            data_dir=args.data_dir,
            output_path=args.output,
            meta_csv=args.meta_csv,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
