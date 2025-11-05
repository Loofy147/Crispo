"""
Example: Deploying and using a Crispo-generated solution in a production environment.

This script shows how to:
1.  Query the Solution Registry to find a previously generated and verified solution.
2.  Retrieve the path to the winning algorithm script.
3.  Execute the script in a simulated production setting to solve a new, unseen problem instance.
"""

import subprocess
import json
import os

def query_solution_registry(project: str, objective: str) -> str | None:
    """
    Uses the Crispo CLI to query the Solution Registry for the best solution
    for a given project and objective.
    """
    print(f"Querying registry for project '{project}' and objective '{objective}'...")
    command = [
        "crispo",
        "--query-registry",
        "--project", project,
        "--objective", objective
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # The CLI is expected to print the metadata of the best solution as a JSON string.
        best_solution = json.loads(result.stdout)
        print("Found best solution:")
        print(json.dumps(best_solution, indent=2))

        # The path in the registry is relative to the project root.
        return best_solution.get("algorithm_script_path")

    except FileNotFoundError:
        print("\\nError: 'crispo' command not found.")
        print("Please ensure the package is installed and in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"\\nFailed to query registry. Crispo exited with status {e.returncode}:")
        print(e.stderr)
        return None
    except json.JSONDecodeError:
        print(f"\\nError: Could not parse the output from the registry query.")
        print(f"Raw output: {result.stdout}")
        return None

def run_deployed_algorithm(script_path: str, new_data_file: str):
    """
    Executes the deployed algorithm on a new data file.
    """
    if not os.path.exists(script_path):
        print(f"\\nError: Algorithm script not found at '{script_path}'")
        return

    print(f"\\nExecuting deployed algorithm '{script_path}' on new data '{new_data_file}'...")

    # This command depends on the specific interface of the generated script.
    # For this example, we assume it's the ski rental script which reads from a file.
    command = ["python3", script_path, "--live-data", new_data_file]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Algorithm executed successfully!")
        print(f"Decision/Output: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"\\nAlgorithm execution failed with status {e.returncode}:")
        print(e.stderr)

if __name__ == "__main__":
    PROJECT_NAME = "SkiRentalDemo"
    OBJECTIVE = "ski rental"

    # 1. Find the best solution in the registry
    #    (This assumes you have already run the ski_rental_demo.py to generate a solution)
    algorithm_path = query_solution_registry(PROJECT_NAME, OBJECTIVE)

    if algorithm_path:
        # 2. Simulate a new, "live" data scenario
        live_data = "live_ski_rental_instance.csv"
        with open(live_data, "w") as f:
            f.write("cost,decision\\n")
            f.write("10,rent\\n")
            f.write("10,rent\\n")
            f.write("10,rent\\n")
            f.write("10,rent\\n")
            f.write("10,rent\\n") # A scenario where buying is optimal (cost > 50)

        # 3. Run the algorithm to get a decision
        run_deployed_algorithm(algorithm_path, live_data)
