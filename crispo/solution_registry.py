"""
Crispo Solution Registry
This module contains the functions for the Solution Registry.
"""

import os
import json
import time
from typing import Dict, Any, List

def save_solution(problem_type: str, performance_metrics: Dict[str, Any]):
    """
    Saves a generated solution package to the registry, with path traversal
    protection and a robust locking mechanism to prevent race conditions.

    Args:
        problem_type (str): The type of problem (e.g., 'ski_rental').
        performance_metrics (Dict[str, Any]): The performance metrics from the Verifier.
    """
    # Security: Sanitize the problem_type to prevent path traversal attacks
    sanitized_problem_type = os.path.basename(problem_type)

    registry_path = "solution_registry"
    problem_path = os.path.join(registry_path, sanitized_problem_type)
    os.makedirs(problem_path, exist_ok=True)

    lock_file_path = os.path.join(problem_path, ".lock")
    lock_fd = None
    attempts = 0
    max_attempts = 50  # 50 * 0.1s = 5 seconds timeout
    lock_acquired = False

    while attempts < max_attempts:
        try:
            # Acquire an exclusive lock using an atomic file creation operation
            lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL)
            lock_acquired = True
            break
        except FileExistsError:
            # If the lock file already exists, wait and retry
            attempts += 1
            time.sleep(0.1)

    if not lock_acquired:
        print("  [Registry] FATAL: Could not acquire lock after multiple attempts. Aborting save.")
        return

    try:
        # --- CRITICAL SECTION ---
        # Find the next version number
        versions = [d for d in os.listdir(problem_path) if d.startswith('v') and os.path.isdir(os.path.join(problem_path, d))]
        next_version = f"v{len(versions) + 1}"
        version_path = os.path.join(problem_path, next_version)
        os.makedirs(version_path)

        # Save the metadata
        metadata_path = os.path.join(version_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(performance_metrics, f, indent=4)

        # Move the generated scripts into the versioned folder
        if os.path.exists("generated_algorithm.py"):
            os.rename("generated_algorithm.py", os.path.join(version_path, "algorithm.py"))
        if os.path.exists("generated_predictor.py"):
            os.rename("generated_predictor.py", os.path.join(version_path, "predictor.py"))

        print(f"  [Registry] Saved {sanitized_problem_type} solution {next_version} to registry.")
        # --- END CRITICAL SECTION ---

    finally:
        # Release the lock
        if lock_fd is not None:
            os.close(lock_fd)
            # Ensure the lock file is removed even if the process is interrupted
            if os.path.exists(lock_file_path):
                os.remove(lock_file_path)

def load_latest_solution(problem_type: str) -> Dict[str, str]:
    """
    Loads the paths to the latest version of a solution for a given problem.

    Args:
        problem_type (str): The type of problem to load.

    Returns:
        Dict[str, str]: A dictionary with paths to the algorithm and predictor scripts.
    """
    problem_path = os.path.join("solution_registry", problem_type)
    if not os.path.exists(problem_path):
        return {}

    versions = [d for d in os.listdir(problem_path) if d.startswith('v')]
    if not versions:
        return {}

    latest_version = sorted(versions, key=lambda v: int(v[1:]))[-1]
    version_path = os.path.join(problem_path, latest_version)

    return {
        "algorithm_script_path": os.path.join(version_path, "algorithm.py"),
        "predictor_script_path": os.path.join(version_path, "predictor.py")
    }

def query_registry(metric: str, threshold: float) -> List[Dict[str, Any]]:
    """
    Queries the registry for solutions that meet a performance criterion.

    Args:
        metric (str): The performance metric to query (e.g., 'competitive_ratio').
        threshold (float): The performance threshold to meet.

    Returns:
        List[Dict[str, Any]]: A list of solutions that meet the criterion.
    """
    results = []
    registry_path = "solution_registry"
    if not os.path.exists(registry_path):
        return []

    for problem_type in os.listdir(registry_path):
        problem_path = os.path.join(registry_path, problem_type)
        for version in os.listdir(problem_path):
            metadata_path = os.path.join(problem_path, version, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if metadata.get(metric, float('inf')) <= threshold:
                    results.append({
                        "problem_type": problem_type,
                        "version": version,
                        "metrics": metadata
                    })
    return results
