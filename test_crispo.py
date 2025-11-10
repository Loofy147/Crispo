"""Unit and integration tests for the Crispo system.

This test suite is organized into classes, each targeting a specific
component of the Crispo system, such as the Verifier and CodeGenerator.
An integration test class is also included to verify the
end-to-end functionality of the main orchestrator pipeline.
"""
import unittest
from unittest import mock
import os
import subprocess
import json
import tempfile
from crispo.crispo_core import (
    CodeGenerator,
    Verifier,
    SkiRentalContext,
    OneMaxSearchContext,
    LayerParameters
)
from crispo.crispo import Crispo, OrchestrationContext
from crispo.predictor_evaluator import PredictorEvaluator

class TestPredictorEvaluator(unittest.TestCase):
    """Tests for the PredictorEvaluator component."""

    def test_uq_calibration(self):
        """Test the UQ calibration evaluation."""
        class MockTimeSeriesPredictor:
            def predict_interval(self, steps=1):
                # Always predict a fixed interval for testing
                return [10, 20]

        evaluator = PredictorEvaluator()
        predictor = MockTimeSeriesPredictor()
        # Test data: two values are inside the [10, 20] interval, one is outside.
        test_data = [(0, 15), (1, 25), (2, 18)]

        metrics = evaluator.evaluate_uq_calibration(predictor, test_data)
        self.assertAlmostEqual(metrics['coverage_rate'], 2/3)
        self.assertAlmostEqual(metrics['interval_sharpness'], 10.0)

class TestVerifier(unittest.TestCase):
    """Tests for the Verifier component."""
    def setUp(self):
        """Initialize the Verifier."""
        self.verifier = Verifier()

    def test_verify_script_syntax_error(self):
        """Check that the verifier correctly identifies syntax errors."""
        script = "def a: pass"
        metrics = self.verifier.verify_script(script)
        self.assertEqual(metrics['syntax_ok'], 0.0)
        self.assertEqual(metrics['runtime_ok'], 0.0)
        self.assertEqual(metrics['overall_quality'], 0.0)

    def test_verify_script_runtime_error(self):
        """Check that the verifier correctly identifies runtime errors."""
        script = "a = 1 / 0"
        metrics = self.verifier.verify_script(script)
        self.assertEqual(metrics['syntax_ok'], 1.0)
        self.assertEqual(metrics['runtime_ok'], 0.0)
        self.assertEqual(metrics['overall_quality'], 0.5)

    def test_verify_script_success(self):
        """Check that the verifier correctly identifies a successful script."""
        script = "a = 1 + 1"
        metrics = self.verifier.verify_script(script)
        self.assertEqual(metrics['syntax_ok'], 1.0)
        self.assertEqual(metrics['runtime_ok'], 1.0)
        self.assertEqual(metrics['overall_quality'], 1.0)

    def test_verify_script_timeout(self):
        """Check that the verifier correctly handles a script that times out."""
        # This script will run indefinitely, so it should be terminated by the timeout.
        script = "while True: pass"
        metrics = self.verifier.verify_script(script)
        # A timeout is a form of runtime failure.
        self.assertEqual(metrics['syntax_ok'], 1.0)
        self.assertEqual(metrics['runtime_ok'], 0.0)
        self.assertEqual(metrics['overall_quality'], 0.5)

    @mock.patch('crispo.crispo_core.save_solution')
    def test_laa_evaluation_saves_correct_problem_type(self, mock_save_solution):
        """Verify that LAA evaluation saves solutions with the correct problem type."""
        with mock.patch('crispo.crispo_core.subprocess.run') as mock_run, \
             mock.patch('pandas.read_csv') as mock_read_csv, \
             mock.patch('crispo.predictor_evaluator.PredictorEvaluator.evaluate_uq_calibration', return_value={}):

            mock_run.return_value = mock.Mock(returncode=0, stdout='{"competitive_ratio": 1.5}')

            # Configure the mock dataframe to behave as expected for the train-test split logic
            mock_series = mock.MagicMock()
            mock_series.iloc.__getitem__.return_value.items.return_value = [] # .iloc[...].items()

            mock_df = mock.MagicMock()
            mock_df.__len__.return_value = 20
            mock_df.iloc.__getitem__.return_value = mock_df
            mock_df.__getitem__.return_value = mock_series
            mock_df.to_csv = mock.Mock() # Mock the to_csv call
            mock_read_csv.return_value = mock_df

            dummy_predictor_code = "class Predictor:\\n    def __init__(self, historical_data_path): pass\\n    def predict_interval(self, steps=1): return [1, 2]".replace('\\n', '\n')
            with open("dummy_predictor.py", "w") as f:
                f.write(dummy_predictor_code)

            with open("dummy_algorithm.py", "w") as f: f.write("pass")
            with open("dummy_history.csv", "w") as f: f.write("value\\n1")

            # The function under test creates and removes this file. Since our mock df doesn't
            # create it, we must create it here to prevent a FileNotFoundError on removal.
            with open("temp_train_history.csv", "w") as f: f.write("value\n1")

            verifier = Verifier()
            context = SkiRentalContext(historical_data_path="dummy_history.csv")

            try:
                verifier.evaluate_learning_augmented_algorithm(
                    algorithm_script_path="dummy_algorithm.py",
                    predictor_script_path="dummy_predictor.py",
                    trust_parameter=0.8,
                    problem_context=context
                )

                mock_save_solution.assert_called_once()
                _, kwargs = mock_save_solution.call_args
                self.assertEqual(kwargs.get('problem_type'), 'ski_rental')
            finally:
                # Clean up all dummy files
                os.remove("dummy_algorithm.py")
                os.remove("dummy_predictor.py")
                os.remove("dummy_history.csv")
                if os.path.exists("temp_train_history.csv"):
                    os.remove("temp_train_history.csv")


class TestSecurity(unittest.TestCase):
    """Tests for security vulnerabilities and fixes."""

    def tearDown(self):
        """Clean up any artifacts created during security tests."""
        if os.path.exists("malicious_file.txt"):
            os.remove("malicious_file.txt")
        if os.path.exists("solution_registry"):
            import shutil
            shutil.rmtree("solution_registry")

    def test_path_traversal_vulnerability(self):
        """Verify that the save_solution function prevents path traversal."""
        from crispo.solution_registry import save_solution

        # Attempt to use a malicious problem_type to write a file outside the registry
        malicious_problem_type = "../malicious_file"
        save_solution(malicious_problem_type, {"metric": 1.0})

        # Check that the file was NOT created in the root directory
        self.assertFalse(os.path.exists("malicious_file.txt"))

        # Check that a sanitized directory was created within the registry instead
        self.assertTrue(os.path.exists("solution_registry/malicious_file"))

    @unittest.skipIf(os.name == 'nt', "Resource limits are not supported on Windows")
    def test_resource_exhaustion_vulnerability(self):
        """Verify that the Verifier terminates resource-intensive scripts."""
        verifier = Verifier()

        # This script attempts to allocate a large amount of memory
        memory_bomb_script = "a = ' ' * (200 * 1024 * 1024) # Attempt to allocate 200MB"
        metrics = verifier.verify_script(memory_bomb_script)

        # The script should fail with a MemoryError, resulting in a runtime failure.
        self.assertEqual(metrics['runtime_ok'], 0.0)
        self.assertIn('error', metrics)
        self.assertIn('memoryerror', metrics['error'].lower())

        # This script will run in an infinite loop, consuming CPU.
        cpu_bomb_script = "while True: pass"
        metrics = verifier.verify_script(cpu_bomb_script)

        # The script should be terminated by the CPU time limit.
        self.assertEqual(metrics['runtime_ok'], 0.0)
        self.assertIn('error', metrics)
        # The process is killed by a signal, not a subprocess timeout.
        self.assertIn('resource limit exceeded', metrics['error'].lower())


class TestCodeGenerator(unittest.TestCase):
    """Tests for the refactored, simplified CodeGenerator."""

    def setUp(self):
        """Set up the CodeGenerator."""
        self.cg = CodeGenerator()

    def test_generate_api_template(self):
        """Verify that 'api' intent generates the API template."""
        script = self.cg.generate("fetch data from api")
        self.assertIn("import requests", script)

    def test_generate_transform_template(self):
        """Verify that 'transform' intent generates the pandas template."""
        script = self.cg.generate("transform the data")
        self.assertIn("import pandas as pd", script)

    def test_generate_numpy_template(self):
        """Verify that 'analyze' intent generates the numpy template."""
        script = self.cg.generate("analyze the numbers")
        self.assertIn("import numpy as np", script)

    def test_generate_simple_template_as_fallback(self):
        """Verify that a generic objective generates the simple template."""
        script = self.cg.generate("do something")
        # Check that the default parameters are embedded
        self.assertIn("weight = 1.00", script)


class TestIntegration(unittest.TestCase):
    """End-to-end tests for the refactored Crispo system."""

    def test_orchestrate_and_execute_e2e_pipeline(self):
        """
        Tests the full, un-mocked pipeline:
        1. Orchestrate a multi-layer pipeline.
        2. Execute the generated scripts in sequence using the Verifier.
        3. Verify that the final output is correct.
        """
        # 1. Orchestrate the pipeline
        objective = "fetch from an api, then transform with pandas"
        context = OrchestrationContext(project="E2E_Test", objective=objective)
        crispo = Crispo(context, problem_context=None)

        # The new orchestrate method generates and returns the scripts
        script_codes = crispo.orchestrate(trust_parameter=0.8)

        # The new design generates a single script based on the most specific keyword.
        self.assertEqual(len(script_codes), 1)

        # 2. Execute the pipeline
        verifier = Verifier()
        # We wrap the single script in a list for verify_pipeline
        result = verifier.verify_pipeline(script_codes)

        # 3. Verify the results
        self.assertEqual(result['overall_quality'], 1.0)
        final_data = result['output']['data']

        # The CodeGenerator will select the API template because 'api' and 'fetch'
        # are checked before 'transform'. The default API call returns a single
        # JSON object, which the template wraps in a list.
        self.assertIsInstance(final_data, list)
        self.assertEqual(len(final_data), 1)
        self.assertIn('userId', final_data[0])


class TestSolutionRegistry(unittest.TestCase):
    """Tests for the Solution Registry functionality."""

    def setUp(self):
        """Set up a clean registry for each test."""
        import shutil
        if os.path.exists("solution_registry"):
            shutil.rmtree("solution_registry")
        # Create dummy files required by save_solution
        with open("generated_algorithm.py", "w") as f: f.write("pass")
        with open("generated_predictor.py", "w") as f: f.write("pass")


    def tearDown(self):
        """Clean up the registry and dummy files after each test."""
        import shutil
        if os.path.exists("solution_registry"):
            shutil.rmtree("solution_registry")
        if os.path.exists("generated_algorithm.py"):
            os.remove("generated_algorithm.py")
        if os.path.exists("generated_predictor.py"):
            os.remove("generated_predictor.py")

    def test_save_and_load_solution(self):
        """Test that a solution can be saved and then loaded."""
        from crispo.solution_registry import save_solution, load_latest_solution

        save_solution("test_problem", {"competitive_ratio": 1.2})

        loaded = load_latest_solution("test_problem")
        self.assertIn("algorithm_script_path", loaded)
        self.assertTrue(os.path.exists(loaded["algorithm_script_path"]))

    def test_query_registry(self):
        """Test the querying functionality of the registry."""
        from crispo.solution_registry import save_solution, query_registry

        save_solution("query_problem", {"competitive_ratio": 1.5})
        save_solution("query_problem", {"competitive_ratio": 1.1})

        results = query_registry("competitive_ratio", 1.2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['problem_type'], "query_problem")
        self.assertAlmostEqual(results[0]['metrics']['competitive_ratio'], 1.1)

    def test_save_solution_race_condition(self):
        """Verify that the save_solution function is robust against race conditions."""
        from crispo.solution_registry import save_solution
        import multiprocessing

        problem_type = "race_condition_test"

        def worker():
            # Re-create dummy files for the new process
            with open("generated_algorithm.py", "w") as f: f.write("pass")
            with open("generated_predictor.py", "w") as f: f.write("pass")
            save_solution(problem_type, {"metric": multiprocessing.current_process().pid})

        processes = [multiprocessing.Process(target=worker) for _ in range(3)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        problem_path = os.path.join("solution_registry", problem_type)
        versions = [d for d in os.listdir(problem_path) if d.startswith('v')]
        self.assertEqual(len(versions), 3, "The locking mechanism failed to prevent a race condition.")


if __name__ == '__main__':
    unittest.main()
