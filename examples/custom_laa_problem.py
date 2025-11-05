"""
Example: Defining a custom Learning-Augmented Algorithm problem for Crispo.

This script provides a template for advanced users who want to extend Crispo
to solve a new, custom online optimization problem that is not natively supported.

The core idea is to subclass the `ProblemContext` abstract base class
and implement the required methods to define the problem's logic.
"""

from crispo.crispo_core import ProblemContext, Crispo
import os

class MyCustomProblemContext(ProblemContext):
    """
    A user-defined context for a custom 'online knapsack' problem.

    In this hypothetical problem, items arrive one by one, and the algorithm
    must decide whether to add them to a knapsack with a limited weight capacity,
    maximizing the total value without exceeding the capacity.
    """
    def __init__(self, complexity: float, trust_parameter: float):
        super().__init__(complexity, trust_parameter)
        self.problem_name = "online_knapsack"

    def get_evaluation_command(self, script_path: str, scenario: dict) -> list[str]:
        """
        Returns the command to execute the generated algorithm script.
        The command will receive the scenario data (e.g., item sequence) via stdin
        or command-line arguments.
        """
        # The generated script is expected to read the scenario from a file
        # whose path is passed as an argument.
        scenario_file = scenario["item_data_path"]
        return ["python3", script_path, "--scenario-file", scenario_file]

    def get_scenarios(self) -> list[dict]:
        """
        Generates or loads a set of scenarios to test the algorithm against.
        For our knapsack problem, each scenario is a different sequence of items.
        """
        # In a real implementation, you would generate or load diverse scenarios.
        # For this example, we'll create a dummy scenario file.
        scenario_dir = "knapsack_scenarios"
        os.makedirs(scenario_dir, exist_ok=True)
        scenario_file = os.path.join(scenario_dir, "scenario1.csv")
        with open(scenario_file, "w") as f:
            f.write("item_value,item_weight\\n")
            f.write("60,10\\n")
            f.write("100,20\\n")
            f.write("120,30\\n")

        return [{"name": "Scenario1", "item_data_path": scenario_file}]

    def get_code_generation_prompt(self) -> str:
        """
        Returns the natural language prompt for the CodeGenerator.
        This prompt guides the LLM to generate the correct algorithm and predictor
        for the 'online knapsack' problem.
        """
        prompt = f\"\"\"
        Generate a Python solution package for the 'online knapsack' problem.
        The goal is to maximize the total value of items in a knapsack with a capacity of 50.

        1.  **Predictor (`generated_predictor.py`):**
            -   Train a model on `knapsack_history.csv`.
            -   Predict the probability distribution of future item weights.
            -   Output a UQ prediction interval for the expected total weight of the next 10 items.

        2.  **Algorithm (`generated_algorithm.py`):**
            -   Reads the prediction interval from the predictor's output.
            -   Implements a Learning-Augmented Algorithm that uses a trust parameter lambda={self.trust_parameter}
                to balance between a greedy strategy (based on the prediction) and a robust baseline
                (e.g., fractional knapsack).
            -   Reads a scenario of items from a CSV file passed as a command-line argument.
            -   Prints the total value achieved by the algorithm to stdout.
        \"\"\"
        return prompt

def run_crispo_with_custom_context():
    """
    Instantiates and runs the main Crispo orchestrator with the custom context.
    """
    print("Running Crispo with a custom LAA problem context...")

    # 1. Define the custom context
    custom_context = MyCustomProblemContext(complexity=0.6, trust_parameter=0.7)

    # 2. Instantiate Crispo with the required parameters
    # Note: We are running Crispo programmatically, not via the CLI.
    crispo_instance = Crispo(
        project="CustomKnapsack",
        objective="custom online knapsack problem",
        project_type="laa",
        domain="logistics",
        problem_context=custom_context
    )

    # 3. Create dummy historical data for the predictor
    with open("knapsack_history.csv", "w") as f:
        f.write("item_value,item_weight\\n")
        f.write("50,8\\n")
        f.write("110,22\\n")
        f.write("130,31\\n")

    # 4. Orchestrate the co-design process
    try:
        crispo_instance.orchestrate()
        print("\\nCrispo execution with custom context finished successfully.")
    except Exception as e:
        print(f"\\nAn error occurred during Crispo execution: {e}")

if __name__ == "__main__":
    run_crispo_with_custom_context()
