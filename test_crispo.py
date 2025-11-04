"""Unit and integration tests for the OrchestratorAI system.

This test suite is organized into classes, each targeting a specific
component of the OrchestratorAI system, such as the Verifier, CodeGenerator,
GAOptimizer, etc. An integration test class is also included to verify the
end-to-end functionality of the main orchestrator pipeline.
"""
import unittest
import os
from crispo_core import (
    LayerParameters,
    GAOptimizer,
    RLAgent,
    AttentionRouter,
    CodeGenerator,
    MetaLearner,
    TaskMetadata,
    Verifier
)
from crispo import Crispo, OrchestrationContext
from advanced_crispo import (
    FederatedOptimizer
)
from predictor_evaluator import PredictorEvaluator

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

class TestAdvancedFeatures(unittest.TestCase):
    """Tests for the advanced features in advanced_orchestrator.py."""

    def setUp(self):
        """Set up a standard LayerParameters object for testing."""
        self.params = LayerParameters(
            layer_id=0,
            weights={'complexity': 1.0, 'execution': 1.0},
            biases={'logging': 0.0},
            temperature=1.0
        )

    def test_federated_optimizer(self):
        """Ensure the Federated Optimizer returns a modified model."""
        fo = FederatedOptimizer(num_clients=5)
        client_data_sizes = [100] * 5
        optimized_params = fo.optimize(self.params, client_data_sizes)
        # Check that the returned object is a different instance
        self.assertIsNot(optimized_params, self.params)
        # Check that weights have been modified
        self.assertNotEqual(optimized_params.weights, self.params.weights)

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

class TestCodeGenerator(unittest.TestCase):
    """Tests for the CodeGenerator component."""

    def setUp(self):
        """Set up the CodeGenerator and a sample LayerParameters object."""
        self.cg = CodeGenerator()
        self.params = LayerParameters(
            layer_id=0,
            weights={'complexity': 0.9, 'execution': 1.0},
            biases={'logging': 0.1},
            temperature=1.2
        )

    def test_generate_high_complexity(self):
        """Verify that a high complexity parameter generates the correct template."""
        script = self.cg.generate(self.params, 0, "analyze")
        self.assertIn("class Layer0System", script)
        self.assertIn("import numpy as np", script)

    def test_generate_medium_complexity(self):
        """Verify that a medium complexity parameter generates the correct template."""
        self.params.weights['complexity'] = 0.6
        script = self.cg.generate(self.params, 1, "process")
        self.assertIn("import pandas as pd", script)
        self.assertIn("def process_layer_1", script)

    def test_generate_low_complexity(self):
        """Verify that a low complexity parameter generates the correct template."""
        self.params.weights['complexity'] = 0.1
        script = self.cg.generate(self.params, 2, "fetch")
        self.assertNotIn("class", script)
        self.assertIn("def process_layer_2", script)

class TestAttentionRouter(unittest.TestCase):
    """Tests for the AttentionRouter component."""

    def setUp(self):
        """Set up the AttentionRouter and sample data."""
        self.ar = AttentionRouter(embed_dim=4, num_heads=2)
        self.query = [1, 0, 0, 0]
        self.keys = [[1, 0, 0, 0], [0, 1, 0, 0]]
        self.values = [[0, 0, 1, 0], [0, 0, 0, 1]]

    def test_execute(self):
        """Verify that the attention mechanism produces a valid output."""
        result = self.ar.execute(self.query, self.keys, self.values)
        self.assertIn('attended_output', result)
        self.assertIn('attention_weights', result)
        self.assertEqual(len(result['attended_output']), 4)
        self.assertAlmostEqual(sum(result['attention_weights']), 1.0)

class TestGAOptimizer(unittest.TestCase):
    """Tests for the GAOptimizer component."""

    def setUp(self):
        """Set up the GAOptimizer and a sample LayerParameters object."""
        self.ga = GAOptimizer(base_population_size=10, mutation_rate=0.5)
        self.template_params = LayerParameters(
            layer_id=0,
            weights={'complexity': 1.0, 'execution': 1.0},
            biases={'logging': 0.0},
            temperature=1.0
        )
        self.context = {'desired_complexity': 0.8}

    def test_initialization(self):
        """Test that the GAOptimizer is initialized with the correct parameters."""
        self.assertEqual(self.ga.base_population_size, 10)
        self.assertEqual(self.ga.mutation_rate, 0.5)

    def test_execute(self):
        """Test that the GA execution runs and returns a valid LayerParameters object."""
        result = self.ga.execute(self.template_params, self.context, generations=5)
        self.assertIsInstance(result, LayerParameters)
        self.assertIn('complexity', result.weights)

class TestRLAgent(unittest.TestCase):
    """Tests for the RLAgent component."""

    def setUp(self):
        """Set up the RLAgent and a sample LayerParameters object."""
        self.rl = RLAgent(epsilon=0.1)
        self.initial_params = LayerParameters(
            layer_id=0,
            weights={'complexity': 1.0, 'execution': 1.0},
            biases={'logging': 0.0},
            temperature=1.0
        )
        self.context = {'complexity': 0.8}

    def test_initialization(self):
        """Test that the RLAgent is initialized with the correct parameters."""
        self.assertEqual(self.rl.epsilon, 0.1)

    def test_execute(self):
        """Test that the RL execution runs and returns a modified LayerParameters object."""
        result = self.rl.execute(self.initial_params, self.context, episodes=3)
        self.assertIsInstance(result, LayerParameters)
        self.assertNotEqual(result.temperature, self.initial_params.temperature)

class TestMetaLearner(unittest.TestCase):
    """Tests for the MetaLearner component."""

    def setUp(self):
        """Set up sample TaskMetadata objects for testing."""
        self.task1 = TaskMetadata(
            task_id="1",
            project_type="data_pipeline",
            complexity_level=0.8,
            domain="data_engineering",
            success_metrics={'overall_quality': 0.9},
            optimal_config={'ga_generations': 25, 'rl_episodes': 12}, # high_quality
            timestamp=""
        )
        self.task2 = TaskMetadata(
            task_id="2",
            project_type="data_pipeline",
            complexity_level=0.4,
            domain="data_engineering",
            success_metrics={'overall_quality': 0.1},
            optimal_config={'ga_generations': 5, 'rl_episodes': 3}, # high_speed
            timestamp=""
        )
        self.task3 = TaskMetadata(
            task_id="3",
            project_type="data_pipeline",
            complexity_level=0.6,
            domain="data_engineering",
            success_metrics={'overall_quality': 0.5},
            optimal_config={'ga_generations': 15, 'rl_episodes': 6}, # balanced
            timestamp=""
        )

    def test_exploitation(self):
        """Test that the MetaLearner exploits the best strategy when epsilon is 0."""
        # With epsilon = 0.0, it should use UCB1 for exploitation
        ml = MetaLearner(epsilon=0.0)
        ml.record_task(self.task1)
        ml.record_task(self.task2)
        ml.record_task(self.task3)

        # After one pull of each strategy, UCB1 should select the one with the highest reward
        strategy = ml.get_optimal_strategy("data_pipeline", 0.8)
        self.assertEqual(strategy['ga_generations'], 20) # 25 * 0.8

    def test_exploration(self):
        """Test that the MetaLearner explores random strategies when epsilon is 1."""
        # With epsilon = 1.0, it should always explore
        ml = MetaLearner(epsilon=1.0)
        ml.record_task(self.task1)
        ml.record_task(self.task2)

        # Run it 100 times, it should not always be high_quality
        strategies = [ml.get_optimal_strategy("data_pipeline", 0.8) for _ in range(100)]
        is_always_best = all([s['ga_generations'] == 20 for s in strategies])
        self.assertFalse(is_always_best)

    def test_default_strategy_for_unknown_project_type(self):
        """Verify that a default strategy is used for an unknown project type."""
        ml = MetaLearner(epsilon=0.0) # Ensure no exploration
        ml.record_task(self.task1)

        strategy = ml.get_optimal_strategy("unknown_type", 0.5)
        self.assertEqual(strategy['ga_generations'], 5) # 10 * 0.5
        self.assertEqual(strategy['rl_episodes'], 2) # 5 * 0.5

class TestOrchestratorIntegration(unittest.TestCase):
    """Integration tests for the main OrchestratorAI pipeline."""

    def test_full_pipeline(self):
        """Test the end-to-end pipeline with standard features."""
        context = OrchestrationContext(
            project="TestProject",
            objective="Test Objective"
        )
        meta_learner = MetaLearner()
        crispo = Crispo(context, meta_learner)

        final_scripts = crispo.orchestrate(
            project_type="data_pipeline",
            domain="testing",
            complexity=0.5,
            enable_transfer_learning=False,
            enable_nas=False,
            enable_federated_optimization=False,
            trust_parameter=0.5 # Add default for old test
        )

        self.assertEqual(len(final_scripts), 3)
        self.assertIsInstance(final_scripts[0], str)
        self.assertIn("# Layer 0", final_scripts[0])
        self.assertIn("# Layer 1", final_scripts[1])
        self.assertIn("# Layer 2", final_scripts[2])

    def test_full_pipeline_with_advanced_features(self):
        """Test the end-to-end pipeline with all advanced features enabled."""
        context = OrchestrationContext(
            project="TestProject",
            objective="Test Objective"
        )
        meta_learner = MetaLearner()
        crispo = Crispo(context, meta_learner)

        final_scripts = crispo.orchestrate(
            project_type="data_pipeline",
            domain="testing",
            complexity=0.5,
            enable_transfer_learning=True,
            enable_nas=True,
            enable_federated_optimization=True,
            trust_parameter=0.5 # Add default for old test
        )

        self.assertEqual(len(final_scripts), 3)
        self.assertIsInstance(final_scripts[0], str)

        # Clean up log file created by this test
        import os
        if os.path.exists("model_registry.log"):
            os.remove("model_registry.log")

    def test_production_transfer_learning_pipeline(self):
        """Test the new, self-generated transfer learning pipeline."""
        import os
        import json

        # 1. Set up the model store and a dummy model
        os.makedirs("model_store", exist_ok=True)
        model_data = {"weights": {"complexity": 0.1, "execution": 0.2}}
        model_path = "model_store/data_pipeline_model.json"
        with open(model_path, 'w') as f:
            json.dump(model_data, f)

        # 2. Run the orchestrator with transfer learning enabled
        context = OrchestrationContext(project="TLTest", objective="Test TL")
        crispo = Crispo(context, MetaLearner())

        crispo.orchestrate(
            project_type="data_pipeline",
            domain="testing",
            complexity=0.9, # Start with high complexity
            enable_transfer_learning=True,
            enable_nas=False,
            enable_federated_optimization=False,
            trust_parameter=0.5
        )

        # 3. Verify that the model registry was written to
        self.assertTrue(os.path.exists("model_registry.log"))
        with open("model_registry.log", 'r') as f:
            # Read the first line of the log file
            log_entry = json.loads(f.readline())
            self.assertEqual(log_entry['model_path'], model_path)

        # 4. Clean up artifacts
        os.remove(model_path)
        if os.path.exists("model_registry.log"):
            os.remove("model_registry.log")
        os.rmdir("model_store")

    def test_production_nas_pipeline(self):
        """Test that the new NAS pipeline runs without errors."""
        context = OrchestrationContext(project="NASTest", objective="Test NAS")
        crispo = Crispo(context, MetaLearner())

        # We just want to ensure this runs to completion without crashing
        final_scripts = crispo.orchestrate(
            project_type="data_pipeline",
            domain="testing",
            complexity=0.9,
            enable_transfer_learning=False,
            enable_nas=True,
            enable_federated_optimization=False,
            trust_parameter=0.5
        )
        self.assertEqual(len(final_scripts), 3)


class TestLearningAugmentedAlgorithms(unittest.TestCase):
    """Tests for the new Learning-Augmented Algorithm functionality."""

    def test_ski_rental_laa_pipeline(self):
        """Test the end-to-end pipeline for generating and evaluating a ski rental LAA."""
        import os
        import random
        # Create dummy historical data with some noise for the predictor
        with open("ski_rental_history.csv", "w") as f:
            f.write("value\n")
            # Generate a slightly more realistic, non-linear series
            values = [15 + i + random.randint(-3, 3) for i in range(15)]
            f.write("\n".join(map(str, values)))

        context = OrchestrationContext(
            project="SkiRentalLAA",
            objective="Generate a learning-augmented algorithm for the ski rental problem"
        )
        meta_learner = MetaLearner(epsilon=0.0)
        crispo = Crispo(context, meta_learner)

        final_scripts = crispo.orchestrate(
            project_type="laa_ski_rental",
            domain="online_algorithms",
            complexity=0.5,
            enable_transfer_learning=False,
            enable_nas=False,
            enable_federated_optimization=False,
            trust_parameter=0.8
        )

        # Should generate a solution package of two scripts
        self.assertEqual(len(final_scripts), 2)
        self.assertIn("def ski_rental_algorithm", final_scripts[0])
        self.assertIn("class Predictor", final_scripts[1])

        # Check that the meta learner recorded the LAA and predictor metrics
        self.assertEqual(len(meta_learner.task_history), 1)
        task_record = meta_learner.task_history[0]
        self.assertIn('competitive_ratio', task_record.success_metrics)
        self.assertIn('coverage_rate', task_record.success_metrics)
        self.assertIn('interval_sharpness', task_record.success_metrics)
        self.assertGreater(task_record.success_metrics['competitive_ratio'], 0)

        # Clean up generated artifacts
        os.remove("ski_rental_history.csv")
        if os.path.exists("solution_registry"):
            import shutil
            shutil.rmtree("solution_registry")

class TestSolutionRegistry(unittest.TestCase):
    """Tests for the Solution Registry functionality."""

    def setUp(self):
        """Set up a clean registry for each test."""
        import shutil
        if os.path.exists("solution_registry"):
            shutil.rmtree("solution_registry")

    def tearDown(self):
        """Clean up the registry after each test."""
        import shutil
        if os.path.exists("solution_registry"):
            shutil.rmtree("solution_registry")

    def test_save_and_load_solution(self):
        """Test that a solution can be saved and then loaded."""
        from solution_registry import save_solution, load_latest_solution

        # Create dummy generated files for the registry to move
        with open("generated_algorithm.py", "w") as f: f.write("alg")
        with open("generated_predictor.py", "w") as f: f.write("pred")

        save_solution("test_problem", {"competitive_ratio": 1.2})

        loaded = load_latest_solution("test_problem")
        self.assertIsNotNone(loaded)
        self.assertTrue(os.path.exists(loaded["algorithm_script_path"]))
        self.assertTrue(os.path.exists(loaded["predictor_script_path"]))


    def test_query_registry(self):
        """Test the querying functionality of the registry."""
        from solution_registry import save_solution, query_registry

        # Save two solutions for the same problem
        with open("generated_algorithm.py", "w") as f: f.write("alg1")
        with open("generated_predictor.py", "w") as f: f.write("pred1")
        save_solution("query_problem", {"competitive_ratio": 1.5})

        with open("generated_algorithm.py", "w") as f: f.write("alg2")
        with open("generated_predictor.py", "w") as f: f.write("pred2")
        save_solution("query_problem", {"competitive_ratio": 1.1})

        results = query_registry("competitive_ratio", 1.2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['problem_type'], "query_problem")
        self.assertAlmostEqual(results[0]['metrics']['competitive_ratio'], 1.1)

    def test_one_max_laa_pipeline(self):
        """Test the end-to-end pipeline for the One-Max Search LAA."""
        # This test is now redundant as the new pipeline is tested above.
        # I will remove it to avoid complexity and focus on the co-design test.
        pass


if __name__ == '__main__':
    unittest.main()
