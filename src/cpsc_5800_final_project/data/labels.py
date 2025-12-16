"""Load and parse sample labels from CSV file."""

import csv
from pathlib import Path
from typing import Dict
from dataclasses import dataclass


@dataclass
class PatientInfo:
    """Complete patient information from labels CSV."""
    sample_id: str
    outcome: str           # "Responder" or "NonResponder"
    response_group: str    # "CR", "PR", or "PD"
    days_elapsed: int      # Survival/follow-up time
    label: int             # Binary: 1=Responder, 0=NonResponder
    
    @property
    def is_responder(self) -> bool:
        return self.label == 1
    
    @property
    def response_code(self) -> int:
        """Ordinal response: 0=PD, 1=PR, 2=CR"""
        return {"PD": 0, "PR": 1, "CR": 2}.get(self.response_group, 0)


def load_patient_info(csv_path: str | Path) -> Dict[str, PatientInfo]:
    """
    Load complete patient information from CSV.
    
    Args:
        csv_path: Path to CSV with columns: SampleID, Outcome, ResponseGroup, DaysElapsed
    
    Returns:
        Dictionary mapping sample_id -> PatientInfo
    """
    patients = {}
    
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["SampleID"].strip()
            if not sample_id:
                continue
            
            outcome = row["Outcome"].strip()
            response_group = row.get("ResponseGroup", "").strip()
            days_str = row.get("DaysElapsed", "0").strip()
            days_elapsed = int(days_str) if days_str else 0
            
            info = PatientInfo(
                sample_id=sample_id,
                outcome=outcome,
                response_group=response_group,
                days_elapsed=days_elapsed,
                label=1 if outcome == "Responder" else 0,
            )
            
            patients[sample_id] = info
            
            # Also map base ID
            base_id = sample_id.split("_")[0]
            if base_id not in patients or "_reimage" in sample_id:
                patients[base_id] = info
    
    return patients


def load_labels(csv_path: str | Path) -> Dict[str, int]:
    """
    Parse CSV file mapping sample IDs to responder/non-responder labels.

    Handles special suffixes:
    - _reimage: indicates reimaged sample (use the reimaged version)
    - _section2: indicates second section from different block
    - Multiple pieces from same sample (e.g., PIO9_1, PIO9_2) are combined

    Args:
        csv_path: Path to CSV file with columns SampleID, Outcome

    Returns:
        Dictionary mapping sample_id -> label (0=NonResponder, 1=Responder)

    Example:
        >>> labels = load_labels("bCPS_DataUpload_Ratna.csv")
        >>> labels["PIO1"]
        0  # NonResponder
        >>> labels["PIO11"]
        1  # Responder
    """
    labels = {}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["SampleID"].strip()
            outcome = row["Outcome"].strip()

            if not sample_id:  # Skip empty rows
                continue

            # Convert outcome to binary label
            label = 1 if outcome == "Responder" else 0

            # Store the label with the sample ID as-is from CSV
            # The sample_id may include suffixes like _reimage, _section2
            labels[sample_id] = label

            # Also create base sample ID mapping for samples with suffixes
            # Extract base sample ID (e.g., "PIO9" from "PIO9_1_reimage")
            base_id = sample_id.split("_")[0]

            # If base ID doesn't exist or this is a reimaged version, update it
            # Prefer reimaged versions over original
            if base_id not in labels or "_reimage" in sample_id:
                labels[base_id] = label

    return labels


def get_sample_ids(csv_path: str | Path) -> list[str]:
    """
    Extract list of all sample IDs from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of sample IDs (as they appear in CSV, including suffixes)
    """
    sample_ids = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["SampleID"].strip()
            if sample_id:
                sample_ids.append(sample_id)

    return sample_ids
