"""Remove ambiguous LIDC nodule clusters from the annotation dataset."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "annotations_all.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
NUDULE_KEY = ["patient_id", "series_uid", "nodule_id"]


def main():
    annotations = pd.read_csv(INPUT_PATH)

    ambiguous_mask = annotations["n_readers_in_cluster"] > 4

    excluded_nodules = (
        annotations.loc[ambiguous_mask]
        .groupby(NUDULE_KEY, as_index=False)
        .agg(
            scan_id=("scan_id", "first"),
            cluster_size=("n_readers_in_cluster", "first"),
        )
    )
    excluded_nodules["reason"] = "cluster_has_more_than_four_annotations"

    clean_annotations = annotations.loc[~ambiguous_mask].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_annotations.to_csv(OUTPUT_DIR / "lidc_annotations_clean.csv", index=False)
    excluded_nodules.to_csv(OUTPUT_DIR / "lidc_excluded_nodules.csv", index=False)

    n_nodules = clean_annotations.groupby(NUDULE_KEY).ngroups
    print(f"Excluded nodules: {len(excluded_nodules)}")
    print(f"Clean nodules: {n_nodules}")
    print(f"Clean annotation rows: {len(clean_annotations)}")


if __name__ == "__main__":
    main()
