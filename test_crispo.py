"""
Unit and Integration Tests for the Crispo System
"""

import unittest
import os
import shutil
from unittest import mock
from crispo.crispo import Crispo, OrchestrationContext
from crispo.crispo_core import CodeGenerator, Verifier, SkiRentalContext, OneMaxSearchContext
from crispo.solution_registry import save_solution, load_latest_solution, query_registry

class TestCodeGenerator(unittest.TestCase):
    """Tests the simplified CodeGenerator."""

    def setUp(self):
        self.generator = CodeGenerator()

    def test_generate_simple_template(self):
        script = self.generator.generate(0, "generate a simple script", 0.1)
        self.assertIn("# Layer 0: Simple Processing", script)

    def test_generate_api_template(self):
        script = self.generator.generate(0, "fetch data from an api", 0.3)
        self.assertIn("# Layer 0: API Interaction", script)

    def test_generate_transform_template(self):
        script = self.generator.generate(1, "transform the data", 0.5)
        self.assertIn("# Layer 1: Data Transformation", script)

    def test_generate_high_complexity_template(self):
        script = self.generator.generate(2, "run a complex analysis", 0.9)
        self.assertIn("# Layer 2: High-Complexity Data Processing", script)

    def test_generate_ski_rental_laa_template(self):
        script = self.generator.generate(0, "solve the ski rental problem", 0.8)
        self.assertIn("Learning-Augmented Algorithm for the Ski Rental Problem", script)


class TestVerifier(unittest.TestCase):
    """Tests the script and pipeline Verifier."""

    def setUp(self):
        self.verifier = Verifier()

    def test_verify_script_success(self):
        script = "a = 1 + 1"
        metrics = self.verifier.verify_script(script)
        self.assertEqual(metrics['overall_quality'], 1.0)

    def test_verify_script_syntax_error(self):
        script = "a = 1 +"
        metrics = self.verifier.verify_script(script)
        self.assertEqual(metrics['overall_quality'], 0.0)
        self.assertEqual(metrics['syntax_ok'], 0.0)

    def test_verify_script_runtime_error(self):
        script = "a = 1 / 0"
        metrics = self.verifier.verify_script(script)
        self.assertEqual(metrics['overall_quality'], 0.5)
        self.assertEqual(metrics['runtime_ok'], 0.0)

    def test_verify_pipeline_success(self):
        scripts = ["import json\nprint(json.dumps({'data': 1}))", "import json\nprint(json.dumps({'data': 2}))"]
        metrics = self.verifier.verify_pipeline(scripts)
        self.assertEqual(metrics['overall_quality'], 1.0)

    def test_verify_pipeline_failure(self):
        scripts = ["import json\nprint(json.dumps({'data': 1}))", "import sys\nsys.exit(1)"]
        metrics = self.verifier.verify_pipeline(scripts)
        self.assertEqual(metrics['overall_quality'], 0.5)

class TestCrispo(unittest.TestCase):
    """Integration tests for the simplified Crispo engine."""

    def setUp(self):
        self.context = OrchestrationContext(project="TestProject", objective="Test Objective")
        # Ensure a clean slate for tests that interact with the file system
        if os.path.exists("solution_registry"):
            shutil.rmtree("solution_registry")
        if os.path.exists("ski_rental_history.csv"):
            os.remove("ski_rental_history.csv")

    def tearDown(self):
        # Clean up any created files
        if os.path.exists("solution_registry"):
            shutil.rmtree("solution_registry")
        if os.path.exists("generated_algorithm.py"):
            os.remove("generated_algorithm.py")
        if os.path.exists("generated_predictor.py"):
            os.remove("generated_predictor.py")
        if os.path.exists("ski_rental_history.csv"):
            os.remove("ski_rental_history.csv")

    def test_orchestrate_simple_pipeline(self):
        """Test the full orchestration of a simple, 3-layer pipeline."""
        crispo = Crispo(self.context)
        scripts = crispo.orchestrate(complexity=0.5, trust_parameter=0.8)
        self.assertEqual(len(scripts), 3)
        self.assertIn("# Layer 0", scripts[0])
        self.assertIn("# Layer 1", scripts[1])
        self.assertIn("# Layer 2", scripts[2])

    @mock.patch('crispo.crispo_core.subprocess.run')
    def test_orchestrate_laa_pipeline(self, mock_run):
        """Test the orchestration of a Learning-Augmented Algorithm package."""
        # Create dummy history file
        with open("ski_rental_history.csv", "w") as f:
            f.write("value\n10\n15\n12")

        # Mock the subprocess calls for the predictor and algorithm
        mock_pred_result = mock.Mock()
        mock_pred_result.returncode = 0
        mock_pred_result.stdout = "[10, 15]"

        mock_alg_result = mock.Mock()
        mock_alg_result.returncode = 0
        mock_alg_result.stdout = '{"competitive_ratio": 1.2, "algorithm_cost": 12, "optimal_cost": 10}'

        mock_run.side_effect = [mock_pred_result, mock_alg_result]

        problem_context = SkiRentalContext()
        crispo = Crispo(
            OrchestrationContext(project="LAATest", objective="solve ski rental"),
            problem_context=problem_context
        )

        scripts = crispo.orchestrate(complexity=0.8, trust_parameter=0.5)

        self.assertEqual(len(scripts), 2)
        # Check that the solution was saved and files were moved
        self.assertTrue(os.path.exists("solution_registry/ski_rental/v1/metadata.json"))
        self.assertTrue(os.path.exists("solution_registry/ski_rental/v1/algorithm.py"))
        self.assertTrue(os.path.exists("solution_registry/ski_rental/v1/predictor.py"))

class TestSolutionRegistry(unittest.TestCase):
    """Tests for the solution registry functionality."""

    def setUp(self):
        # Create a dummy registry for testing
        self.registry_path = "solution_registry"
        if os.path.exists(self.registry_path):
            shutil.rmtree(self.registry_path)
        os.makedirs(os.path.join(self.registry_path, "test_problem", "v1"))

    def tearDown(self):
        # Clean up the dummy registry
        if os.path.exists(self.registry_path):
            shutil.rmtree(self.registry_path)
        if os.path.exists("generated_algorithm.py"):
            os.remove("generated_algorithm.py")
        if os.path.exists("generated_predictor.py"):
            os.remove("generated_predictor.py")

    def test_save_and_load_solution(self):
        """Test that a solution can be saved and then loaded."""
        # Create dummy solution files
        with open("generated_algorithm.py", "w") as f: f.write("pass")
        with open("generated_predictor.py", "w") as f: f.write("pass")

        metrics = {"competitive_ratio": 1.5}
        save_solution("test_problem", metrics)

        loaded = load_latest_solution("test_problem")
        self.assertIn("algorithm_script_path", loaded)
        self.assertTrue(os.path.exists(loaded["algorithm_script_path"]))

    def test_query_registry(self):
        """Test the querying functionality of the registry."""
        # Create dummy solution files and metadata
        with open("generated_algorithm.py", "w") as f: f.write("pass")
        with open("generated_predictor.py", "w") as f: f.write("pass")
        save_solution("query_problem", {"competitive_ratio": 1.2, "coverage": 0.9})

        with open("generated_algorithm.py", "w") as f: f.write("pass")
        with open("generated_predictor.py", "w") as f: f.write("pass")
        save_solution("query_problem", {"competitive_ratio": 1.8, "coverage": 0.8})

        results = query_registry("competitive_ratio", 1.5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['metrics']['competitive_ratio'], 1.2)


if __name__ == '__main__':
    unittest.main()
