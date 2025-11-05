"""
Example: Using Crispo to solve the Ski Rental problem.

This script demonstrates how to use the Crispo command-line interface
to generate a solution for the classic ski rental online optimization problem.
"""

import subprocess

def run_crispo_ski_rental():
    """
    Calls the Crispo CLI to generate a Learning-Augmented Algorithm
    for the Ski Rental problem.
    """
    command = [
        "crispo",
        "--project", "SkiRentalDemo",
        "--objective", "ski rental",
        "--project_type", "laa",
        "--domain", "finance",
        "--complexity", "0.5",
        "--save-metaknowledge", "ski_rental_metaknowledge.pkl"
    ]

    print(f"Running command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("\nCrispo execution successful!")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
    except FileNotFoundError:
        print("\\nError: 'crispo' command not found.")
        print("Please make sure you have installed the package correctly, e.g., by running 'pip install .'")
    except subprocess.CalledProcessError as e:
        print("\\nCrispo execution failed.")
        print(f"Return code: {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)

if __name__ == "__main__":
    # Note: To run this demo, you need a 'ski_rental_history.csv' file in the root directory.
    # Creating a dummy file for demonstration purposes.
    try:
        with open("ski_rental_history.csv", "w") as f:
            f.write("cost,decision\\n")
            f.write("10,rent\\n")
            f.write("10,rent\\n")
            f.write("100,buy\\n")
    except IOError as e:
        print(f"Could not create dummy ski_rental_history.csv: {e}")

    run_crispo_ski_rental()
