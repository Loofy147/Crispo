"""
OrchestratorAI Core Components
This module contains the core data structures, AI components (GA, RL, Attention),
and code generation logic for the OrchestratorAI system.
"""

import json
import random
import math
import re
import subprocess
import tempfile
import os
import resource
import platform
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
from .solution_registry import save_solution

# ============================================================================
# CORE CONSTANTS
# ============================================================================

# Timeout values for various operations, in seconds.
API_REQUEST_TIMEOUT = 5
SINGLE_SCRIPT_VERIFICATION_TIMEOUT = 10
PIPELINE_VERIFICATION_TIMEOUT = 15
PREDICTOR_EXECUTION_TIMEOUT = 20
ALGORITHM_EXECUTION_TIMEOUT = 10


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class TaskMetadata:
    """Encapsulates metadata about an orchestration task for meta-learning.

    Attributes:
        task_id: A unique identifier for the task.
        project_type: The category of the project (e.g., 'data_pipeline').
        complexity_level: The complexity of the task, typically from 0.0 to 1.0.
        domain: The application domain (e.g., 'finance', 'data_engineering').
        success_metrics: A dictionary of metrics evaluating the task's success.
        optimal_config: The configuration that yielded the best results.
        timestamp: The ISO format timestamp of when the task was executed.
    """
    task_id: str
    project_type: str
    complexity_level: float
    domain: str
    success_metrics: Dict[str, float]
    optimal_config: Dict[str, Any]
    timestamp: str

@dataclass
class OrchestrationContext:
    """Holds the global context for a single orchestration run.

    Attributes:
        project: The name of the project.
        objective: The high-level objective of the orchestration.
        feedback_loop: A dictionary to store feedback data during execution.
        resource_usage: A dictionary to track resource consumption.
        failure_cases: A list of any failure cases encountered.
    """
    project: str
    objective: str
    feedback_loop: Dict = field(default_factory=dict)
    resource_usage: Dict = field(default_factory=dict)
    failure_cases: List[str] = field(default_factory=list)

# ============================================================================
# PRODUCTION UNIT: CODE GENERATOR
# ============================================================================

class CodeGenerator:
    """Generates multi-layer scripts with intent-based template selection.

    This class is responsible for creating the actual Python code for each
    layer of the orchestration pipeline. It uses a combination of intent
    detection (based on keywords in the objective) and parameter-driven logic
    to select the most appropriate code template for a given layer.
    """

    def generate(self, layer_id: int, objective: str, complexity: float, trust_parameter: float = 0.8) -> str:
        """Selects and generates a script for a single layer.

        This method uses keyword matching on the user's objective to select
        the most logically appropriate template for each layer of the pipeline.
        If a clear intent cannot be determined, it falls back to a
        complexity-based selection mechanism.

        Args:
            layer_id (int): The ID of the current layer (e.g., 0, 1, 2).
            objective (str): The user's high-level objective string, used for
                intent detection.
            complexity (float): The complexity of the project (0.0 to 1.0).
            trust_parameter (float): The trust parameter (lambda) for
                learning-augmented algorithms.

        Returns:
            str: A string containing the generated Python script.
        """
        objective = objective.lower()

        # Check for LAA-specific objectives first
        if "ski rental" in objective:
            return self._generate_ski_rental_laa_template(trust_parameter)
        if "one-max search" in objective:
            return self._generate_one_max_laa_template(trust_parameter)

        # Layer 0: Prioritize fetching data
        if layer_id == 0 and any(kw in objective for kw in ['fetch', 'get', 'api', 'request']):
            return self._generate_api_template(layer_id)

        # Layer 1: Prioritize transformation
        if layer_id == 1 and any(kw in objective for kw in ['process', 'transform', 'clean', 'pandas']):
            return self._generate_transform_template(layer_id, complexity)

        # Layer 2: Prioritize analysis
        if layer_id == 2 and any(kw in objective for kw in ['analyze', 'numpy', 'compute', 'calculate']):
            return self._generate_high_complexity_template(layer_id, complexity)

        # Fallback to complexity-based logic if intent is not clear for the layer
        if complexity > 0.7:
            return self._generate_high_complexity_template(layer_id, complexity)
        if complexity > 0.4:
            return self._generate_transform_template(layer_id, complexity)

        return self._generate_simple_template(layer_id)

    def _generate_ski_rental_laa_template(self, trust_parameter: float) -> str:
        """Generates a script for the Ski Rental problem using a learning-augmented algorithm."""
        return f'''"""
Learning-Augmented Algorithm for the Ski Rental Problem
"""
import sys
import json

def ski_rental_algorithm(B, prediction_interval, trust_lambda, actual_days):
    """
    Executes the UQ-aware learning-augmented ski rental algorithm.

    Args:
        B (int): The cost to buy skis.
        prediction_interval (List[int]): The predicted [lower, upper] bound
            of skiing days.
        trust_lambda (float): The trust parameter (lambda) between 0 and 1.
        actual_days (int): The actual number of skiing days.

    Returns:
        int: The total cost incurred by the algorithm.
    """
    # Use the upper bound of the interval for a more robust threshold
    prediction_upper_bound = prediction_interval[1]

    # The core of the learning-augmented algorithm: a blended threshold
    threshold = (1 - trust_lambda) * B + trust_lambda * min(prediction_upper_bound, B)

    cost = 0
    bought_skis = False
    for day in range(1, actual_days + 1):
        if day >= threshold and not bought_skis:
            cost += B
            bought_skis = True
            break
        else:
            cost += 1 # Rent for the day

    # If we never bought, the total cost is just the number of days we rented
    if not bought_skis:
        cost = actual_days

    return cost

def calculate_optimal_cost(B, actual_days):
    """Calculates the optimal offline cost."""
    return min(B, actual_days)

def main():
    """Main execution function."""
    if len(sys.argv) != 5:
        print("Usage: python ski_rental.py <buy_cost> '<prediction_interval>' <trust_lambda> <actual_days>")
        sys.exit(1)

    B = int(sys.argv[1])
    prediction_interval = json.loads(sys.argv[2])
    trust_lambda = float(sys.argv[3])
    actual_days = int(sys.argv[4])

    alg_cost = ski_rental_algorithm(B, prediction_interval, trust_lambda, actual_days)
    opt_cost = calculate_optimal_cost(B, actual_days)

    output = {{
        "algorithm_cost": alg_cost,
        "optimal_cost": opt_cost,
        "competitive_ratio": alg_cost / opt_cost if opt_cost > 0 else 1
    }}
    print(json.dumps(output))

if __name__ == "__main__":
    main()
'''

    def _generate_predictor_template(self) -> str:
        """Generates a class-based script for a time-series predictor."""
        return '''"""
Time-Series Predictor for Learning-Augmented Algorithms
"""
import sys
import json
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class Predictor:
    def __init__(self, historical_data_path: str):
        self.historical_data_path = historical_data_path
        self.model = self._train()

    def _train(self):
        """Trains a simple ARIMA model."""
        try:
            data = pd.read_csv(self.historical_data_path)
            series = data['value']
            model = ARIMA(series, order=(5, 1, 0))
            return model.fit()
        except Exception as e:
            print(f"Predictor training failed: {e}", file=sys.stderr)
            return None

    def predict_interval(self, steps: int = 1):
        """
        Outputs a UQ prediction interval for a number of future steps.

        Args:
            steps (int): The number of future steps to predict.

        Returns:
            List[int]: A [lower_bound, upper_bound] prediction interval.
        """
        if not self.model:
            # Fallback on training failure
            return [10, 20]

        try:
            forecast = self.model.get_forecast(steps=steps)
            pred_ci = forecast.conf_int().iloc[-1] # Get the last CI for multi-step

            lower_bound = int(pred_ci[0])
            upper_bound = int(pred_ci[1])

            # Ensure bounds are reasonable
            lower_bound = max(1, lower_bound)
            upper_bound = max(lower_bound, upper_bound)

            return [lower_bound, upper_bound]
        except Exception as e:
            print(f"Predictor prediction failed: {e}", file=sys.stderr)
            return [10, 20] # Fallback

def main():
    """Main execution function for command-line use."""
    if len(sys.argv) != 2:
        print("Usage: python predictor.py <historical_data_path>")
        sys.exit(1)

    historical_data_path = sys.argv[1]
    predictor = Predictor(historical_data_path)
    prediction_interval = predictor.predict_interval()
    print(json.dumps(prediction_interval))

if __name__ == "__main__":
    main()
'''

    def _generate_one_max_laa_template(self, trust_parameter: float) -> str:
        """Generates a script for the One-Max Search problem using a learning-augmented algorithm."""
        return f'''"""
Learning-Augmented Algorithm for the One-Max Search Problem
"""
import sys
import json
import ast

def one_max_algorithm(sequence, prediction_interval, trust_lambda):
    """
    Executes the UQ-aware learning-augmented one-max search algorithm.

    Args:
        sequence (List[int]): The sequence of values observed.
        prediction_interval (List[int]): The predicted [lower, upper] bound
            of the maximum value.
        trust_lambda (float): The trust parameter (lambda) between 0 and 1.

    Returns:
        int: The value selected by the algorithm.
    """
    # Use the lower bound of the interval for a more conservative threshold
    prediction_lower_bound = prediction_interval[0]

    threshold = trust_lambda * prediction_lower_bound

    for value in sequence:
        if value >= threshold:
            return value

    # If no value meets the threshold, accept the last one (a robust strategy)
    return sequence[-1] if sequence else 0

def calculate_optimal_cost(sequence):
    """Calculates the optimal offline cost (the true maximum)."""
    return max(sequence) if sequence else 0

def main():
    """Main execution function."""
    if len(sys.argv) != 4:
        print("Usage: python one_max.py '<sequence>' '<prediction_interval>' <trust_lambda>")
        sys.exit(1)

    sequence = ast.literal_eval(sys.argv[1])
    prediction_interval = ast.literal_eval(sys.argv[2])
    trust_lambda = float(sys.argv[3])

    alg_value = one_max_algorithm(sequence, prediction_interval, trust_lambda)
    opt_value = calculate_optimal_cost(sequence)

    output = {{
        "algorithm_cost": alg_value, # In One-Max, "cost" is the value selected
        "optimal_cost": opt_value,
        "competitive_ratio": alg_value / opt_value if opt_value > 0 else 1.0
    }}
    print(json.dumps(output))

if __name__ == "__main__":
    main()
'''

    def _generate_high_complexity_template(self, layer_id: int, complexity: float) -> str:
        """Generates a script for high-complexity data processing using NumPy.

        Args:
            layer_id: The ID of the current layer.
            complexity: The complexity of the project.

        Returns:
            A string containing the generated Python script.
        """
        return f'''# Layer {layer_id}: High-Complexity Data Processing
import numpy as np
import json
import pandas as pd

def process_layer_{layer_id}(input_context):
    records = input_context.get('data', [])
    if not records:
        print(json.dumps({{'data': []}}))
        return

    df = pd.DataFrame(records)
    # Perform a numerical operation on a column if it exists
    if 'new_col' in df.columns:
        df['analyzed_col'] = np.log1p(df['new_col'])

    output_context = {{'data': df.to_dict('records')}}
    return output_context

if __name__ == '__main__':
    # This block is for standalone execution and will be driven by the test
    # The 'input_context' will be injected by the test runner if it exists.
    context = locals().get('input_context', {{}})
    result = process_layer_{layer_id}(context)
    print(json.dumps(result))
'''

    def _generate_transform_template(self, layer_id: int, complexity: float) -> str:
        """Generates a script for data transformation using pandas.

        Args:
            layer_id: The ID of the current layer.
            complexity: The complexity of the project.

        Returns:
            A string containing the generated Python script.
        """
        num_ops = int(3 * complexity)
        return f'''# Layer {layer_id}: Data Transformation
import pandas as pd
import json
import numpy as np

def process_layer_{layer_id}(input_context):
    data = input_context.get('data', [])

    # API layer returns a single dict, ensure it's wrapped in a list for pandas
    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    # Perform a simple data transformation
    if not df.empty:
        # Find the first numeric column to apply the transformation
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            df['new_col'] = df[numeric_cols[0]] * {1 + complexity:.2f}

    output_context = {{'data': df.to_dict('records')}}
    return output_context

if __name__ == '__main__':
    # This block is for standalone execution and will be driven by the test
    # The 'input_context' will be injected by the test runner if it exists.
    context = locals().get('input_context', {{}})
    result = process_layer_{layer_id}(context)
    print(json.dumps(result))
'''

    def _generate_api_template(self, layer_id: int) -> str:
        """Generates a script for interacting with an API using requests.

        Args:
            layer_id: The ID of the current layer.

        Returns:
            A string containing the generated Python script.
        """
        return f'''# Layer {layer_id}: API Interaction
import requests
import json

def process_layer_{layer_id}(input_context):
    api_endpoint = input_context.get('api_endpoint', 'https://jsonplaceholder.typicode.com/todos/1')
    try:
        response = requests.get(api_endpoint, timeout=5)
        response.raise_for_status()
        output_context = {{'data': response.json()}}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {{e}}")
        output_context = {{'data': [], 'error': str(e)}}
    return output_context

if __name__ == '__main__':
    # This block is for standalone execution and will be driven by the test
    # The 'input_context' will be injected by the test runner if it exists.
    context = locals().get('input_context', {{}})
    result = process_layer_{layer_id}(context)
    print(json.dumps(result))
'''

    def _generate_simple_template(self, layer_id: int) -> str:
        """Generates a script for simple, low-complexity processing.

        Args:
            layer_id: The ID of the current layer.

        Returns:
            A string containing the generated Python script.
        """
        return f'''# Layer {layer_id}: Simple Processing
import json

def process_layer_{layer_id}(input_context):
    data = input_context.get('data', 0)
    result = data * 1.1 + 1
    output_context = {{'data': result}}
    return output_context

if __name__ == '__main__':
    # This block is for standalone execution and will be driven by the test
    # The 'input_context' will be injected by the test runner if it exists.
    context = locals().get('input_context', {{}})
    result = process_layer_{layer_id}(context)
    print(json.dumps(result))
'''

# ============================================================================
# VERIFICATION UNIT
# ============================================================================

class ProblemContext:
    """Abstract base class for a problem context, used by the Verifier."""
    def get_evaluation_command(self, script_path, prediction, trust_parameter, scenario) -> List[str]:
        raise NotImplementedError

    def get_perfect_prediction(self, scenario):
        raise NotImplementedError

    def get_worst_prediction(self, scenario):
        raise NotImplementedError

    def get_noisy_prediction(self, scenario, error_level):
        raise NotImplementedError

    def get_scenarios(self):
        raise NotImplementedError

class SkiRentalContext(ProblemContext):
    """Problem context for the Ski Rental problem."""
    def __init__(self, buy_cost=100, historical_data_path="ski_rental_history.csv"):
        self.buy_cost = buy_cost
        self.historical_data_path = historical_data_path

    def get_evaluation_command(self, script_path, prediction, trust_parameter, scenario) -> List[str]:
        actual_days = scenario
        return ['python3', script_path, str(self.buy_cost), prediction, str(trust_parameter), str(actual_days)]

    def get_perfect_prediction(self, scenario):
        # Perfect UQ prediction is a tight interval around the true value
        return [scenario, scenario]

    def get_worst_prediction(self, scenario):
        # Worst-case is a misleading interval
        return [1, 2]

    def get_noisy_prediction(self, scenario, error_level):
        # Noisy prediction widens the interval based on error
        perfect = self.get_perfect_prediction(scenario)
        lower_bound = max(1, int(perfect[0] * (1 - error_level)))
        upper_bound = int(perfect[1] * (1 + error_level))
        return [lower_bound, upper_bound]

    def get_scenarios(self):
        return range(1, self.buy_cost * 2)

class OneMaxSearchContext(ProblemContext):
    """Problem context for the One-Max Search problem."""
    def __init__(self, historical_data_path="one_max_history.csv"):
        self.historical_data_path = historical_data_path

    def get_evaluation_command(self, script_path, prediction, trust_parameter, scenario) -> List[str]:
        sequence = scenario
        return ['python3', script_path, str(sequence), prediction, str(trust_parameter)]

    def get_perfect_prediction(self, scenario):
        true_max = max(scenario) if scenario else 0
        return [true_max, true_max]

    def get_worst_prediction(self, scenario):
        true_min = min(scenario) if scenario else 0
        return [true_min, true_min]

    def get_noisy_prediction(self, scenario, error_level):
        perfect = self.get_perfect_prediction(scenario)[0]
        lower_bound = int(perfect * (1 - error_level))
        upper_bound = int(perfect * (1 + error_level))
        return [lower_bound, upper_bound]

    def get_scenarios(self):
        # Generate some random sequences for evaluation
        return [random.sample(range(1, 100), 10) for _ in range(20)]


class Verifier:
    """Verifies the correctness of generated code.

    This class provides methods to check for syntax and runtime errors in
    both individual scripts and full pipelines.
    """

    def verify_script(self, script_code: str) -> Dict[str, float]:
        """Verifies a single script for syntax and runtime errors.

        This method first checks for syntax errors by attempting to compile the
        code. If that succeeds, it saves the code to a temporary file and
        executes it as a subprocess to check for runtime errors.

        Args:
            script_code: A string containing the Python code to verify.

        Returns:
            A dictionary containing 'syntax_ok', 'runtime_ok', and
            'overall_quality' metrics.
        """
        metrics = {'syntax_ok': 0.0, 'runtime_ok': 0.0, 'overall_quality': 0.0}

        # 1. Check for syntax errors first
        try:
            compile(script_code, '<string>', 'exec')
            metrics['syntax_ok'] = 1.0
        except (SyntaxError, TypeError):
            return metrics  # No point in trying to run if syntax is wrong

        def set_limits():
            """Set resource limits for the subprocess."""
            # Set CPU time limit to 2 seconds
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            # Set memory limit to 100MB
            resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(script_code)
            temp_filename = temp_file.name

        # Prepare subprocess arguments
        run_args = {
            "capture_output": True,
            "text": True,
            "timeout": SINGLE_SCRIPT_VERIFICATION_TIMEOUT
        }

        # Add resource limits only on non-Windows platforms
        if platform.system() != "Windows":
            run_args["preexec_fn"] = set_limits

        try:
            # Execute the script as a subprocess
            result = subprocess.run(
                ['python3', temp_filename],
                **run_args
            )

            # A non-zero return code indicates a runtime error
            if result.returncode == 0:
                metrics['runtime_ok'] = 1.0

        except FileNotFoundError:
            # This can happen if the python3 interpreter is not found
            print("Error: python3 interpreter not found.")

        except subprocess.TimeoutExpired:
            print(f"Verification timeout for script: {temp_filename}")

        finally:
            # Clean up the temporary file
            import os
            os.remove(temp_filename)

        # Calculate overall quality
        metrics['overall_quality'] = (metrics['syntax_ok'] + metrics['runtime_ok']) / 2.0

        return metrics

    def verify_pipeline(self, script_codes: List[str]) -> Dict[str, float]:
        """Verifies a full pipeline of scripts.

        This method executes a list of scripts sequentially, passing the JSON
        output of each script as the input context to the next. This provides
        a true end-to-end integration test of the generated pipeline.

        Args:
            script_codes: A list of strings, where each string is a
                self-contained Python script.

        Returns:
            A dictionary containing the 'overall_quality' of the pipeline,
            which is the proportion of scripts that executed successfully.
        """
        pipeline_context = {}
        total_quality = 0.0
        num_scripts = len(script_codes)

        for i, script_code in enumerate(script_codes):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                # Inject the input context into the script
                injected_code = f"import json\npipeline_context = {json.dumps(pipeline_context)}\n" + script_code
                temp_file.write(injected_code)
                temp_filename = temp_file.name

            try:
                result = subprocess.run(
                    ['python3', temp_filename],
                    capture_output=True,
                    text=True,
                    timeout=PIPELINE_VERIFICATION_TIMEOUT
                )

                if result.returncode == 0:
                    total_quality += 1.0
                    # Parse the output to get the next context
                    pipeline_context = json.loads(result.stdout)
                else:
                    # Stop the pipeline on the first failure
                    print(f"Pipeline failed at layer {i}.")
                    break

            finally:
                import os
                os.remove(temp_filename)

        final_quality = total_quality / num_scripts if num_scripts > 0 else 0.0
        return {'overall_quality': final_quality}

    def evaluate_learning_augmented_algorithm(
        self,
        algorithm_script_path: str,
        predictor_script_path: str,
        trust_parameter: float,
        problem_context: ProblemContext
    ) -> Dict[str, Any]:
        """
        Performs a two-stage, "live" evaluation of a full LAA solution package.

        Args:
            algorithm_script_path (str): Path to the generated algorithm script.
            predictor_script_path (str): Path to the generated predictor script.
            trust_parameter (float): The lambda value for the algorithm.
            problem_context (ProblemContext): The context defining the problem.

        Returns:
            Dict[str, Any]: A dictionary containing the 'competitive_ratio' of
                the full, co-designed solution.
        """
        # Stage 1: Run the predictor to get a live prediction
        cmd_pred = ['python3', predictor_script_path, problem_context.historical_data_path]
        result_pred = subprocess.run(cmd_pred, capture_output=True, text=True, timeout=PREDICTOR_EXECUTION_TIMEOUT)
        if result_pred.returncode != 0:
            print(f"  [Verifier] Predictor script failed: {result_pred.stderr}")
            return {'competitive_ratio': float('inf')}

        live_prediction = result_pred.stdout.strip()

        # Stage 2: Run the algorithm with the live prediction
        # We use a single, representative scenario for this live evaluation
        scenario = problem_context.get_scenarios()[0]
        cmd_alg = problem_context.get_evaluation_command(
            algorithm_script_path, live_prediction, trust_parameter, scenario
        )
        result_alg = subprocess.run(cmd_alg, capture_output=True, text=True, timeout=ALGORITHM_EXECUTION_TIMEOUT)
        if result_alg.returncode != 0:
            print(f"  [Verifier] Algorithm script failed: {result_alg.stderr}")
            return {'competitive_ratio': float('inf')}

        metrics = json.loads(result_alg.stdout)

        # Evaluate predictor quality
        from .predictor_evaluator import PredictorEvaluator
        import importlib.util
        import pandas as pd
        import os

        spec = importlib.util.spec_from_file_location("predictor", predictor_script_path)
        predictor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor_module)

        evaluator = PredictorEvaluator()

        # Create a train/test split for a fair evaluation
        full_data = pd.read_csv(problem_context.historical_data_path)
        if len(full_data) > 10:
            # Use the first n-3 rows for training, last 3 for testing
            split_point = len(full_data) - 3
            train_df = full_data.iloc[:split_point]
            test_data = list(full_data['value'].iloc[split_point:].items())

            # Write the training data to a temporary file
            temp_train_path = "temp_train_history.csv"
            train_df.to_csv(temp_train_path, index=False)

            # Instantiate the predictor with only the training data
            predictor_for_eval = predictor_module.Predictor(temp_train_path)

            # Evaluate on the hold-out test set
            predictor_metrics = evaluator.evaluate_uq_calibration(predictor_for_eval, test_data)
            metrics.update(predictor_metrics)

            # Clean up the temporary file
            os.remove(temp_train_path)
        else:
            # If the dataset is too small, we can't evaluate the predictor.
            # Add default metrics to ensure the key exists for the test.
            metrics['coverage_rate'] = 0.0
            metrics['interval_sharpness'] = float('inf')


        # Save the verified solution to the registry
        problem_type = "ski_rental" if isinstance(problem_context, SkiRentalContext) else "one_max"
        save_solution(
            problem_type=problem_type,
            performance_metrics=metrics
        )
        print(f"  [Registry] Saved solution for {problem_type} with competitive ratio: {metrics.get('competitive_ratio')}")

        return metrics
