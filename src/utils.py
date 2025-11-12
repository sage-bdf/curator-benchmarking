"""Utility functions for the benchmarking framework."""
import shutil
from pathlib import Path
from typing import List, Optional


def organize_existing_task_files(
    source_dir: Path,
    task_name: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Organize existing task files into the proper structure.
    
    This is a helper function to organize files that are already in the tasks/
    directory but not yet organized into task folders.
    
    Args:
        source_dir: Directory containing the task files
        task_name: Name for the task
        output_dir: Where to create the task directory (default: tasks/)
    
    Returns:
        Path to the created task directory
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "tasks"
    
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV and TSV files
    csv_files = list(source_dir.glob("*.csv"))
    tsv_files = list(source_dir.glob("*.tsv"))
    all_files = csv_files + tsv_files
    
    # Identify ground truth files
    ground_truth_files = [
        f for f in all_files
        if "ground" in f.name.lower() and "truth" in f.name.lower()
    ]
    
    # Identify input files (everything else)
    input_files = [f for f in all_files if f not in ground_truth_files]
    
    # Copy ground truth files
    for gt_file in ground_truth_files:
        dest = task_dir / "ground_truth" / gt_file.suffix[1:]  # .csv or .tsv
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(gt_file, dest)
    
    # Copy input files (use first one as primary input)
    if input_files:
        primary_input = input_files[0]
        dest = task_dir / f"input_data{primary_input.suffix}"
        shutil.copy2(primary_input, dest)
    
    # Create a default prompt if none exists
    prompt_file = task_dir / "default_prompt.txt"
    if not prompt_file.exists():
        prompt_file.write_text(
            "Please process the following metadata according to the task requirements.\n"
            "Return the result as JSON with the same structure as the input.\n"
        )
    
    print(f"Organized task files into: {task_dir}")
    return task_dir

