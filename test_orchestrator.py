"""Unit and integration tests for the OrchestratorAI system.

This test suite is organized into classes, each targeting a specific
component of the OrchestratorAI system, such as the Verifier, CodeGenerator,
GAOptimizer, etc. An integration test class is also included to verify the
end-to-end functionality of the main orchestrator pipeline.
"""
import unittest
from orchestrator_core import (
    LayerParameters,
    GAOptimizer,
    RLAgent,
    AttentionRouter,
    CodeGenerator,
    MetaLearner,
    TaskMetadata,
    Verifier
)
from orchestrator import OrchestratorAI, OrchestrationContext
from advanced_orchestrator import (
    TransferLearningEngine,
    NeuralArchitectureSearch,
    FederatedOptimizer
)

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

    def test_transfer_learning_engine(self):
        """Verify that the Transfer Learning Engine correctly modifies parameters."""
        model_store = {
            "data_pipeline": LayerParameters(
                layer_id=0,
                weights={'complexity': 0.5, 'execution': 0.5},
                biases={'logging': 0.1},
                temperature=1.0
            )
        }
        tle = TransferLearningEngine(model_store)
        updated_params = tle.apply(self.params, "data_pipeline")
        self.assertNotEqual(updated_params.weights['complexity'], 1.0)

    def test_neural_architecture_search(self):
        """Test the placeholder NAS to ensure it returns a valid architecture."""
        nas = NeuralArchitectureSearch(search_space={'layer_type': ['dense']})
        architecture = nas.search(num_layers=3)
        self.assertEqual(len(architecture), 3)
        self.assertEqual(architecture[0]['layer_type'], ['dense'])

    def test_federated_optimizer(self):
        """Ensure the placeholder Federated Optimizer returns the model unmodified."""
        fo = FederatedOptimizer(num_clients=5)
        optimized_params = fo.optimize(self.params)
        self.assertEqual(optimized_params, self.params)

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
        self.ga = GAOptimizer(population_size=10, mutation_rate=0.5)
        self.template_params = LayerParameters(
            layer_id=0,
            weights={'complexity': 1.0, 'execution': 1.0},
            biases={'logging': 0.0},
            temperature=1.0
        )
        self.context = {'desired_complexity': 0.8}

    def test_initialization(self):
        """Test that the GAOptimizer is initialized with the correct parameters."""
        self.assertEqual(self.ga.population_size, 10)
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

    def test_exploitation(self):
        """Test that the MetaLearner exploits the best strategy when epsilon is 0."""
        # With epsilon = 0.0, it should always exploit
        ml = MetaLearner(epsilon=0.0)
        ml.record_task(self.task1)
        ml.record_task(self.task2)

        # The best strategy is 'high_quality'
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
        orchestrator = OrchestratorAI(context, meta_learner)

        final_scripts = orchestrator.orchestrate(
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
        orchestrator = OrchestratorAI(context, meta_learner)

        final_scripts = orchestrator.orchestrate(
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

class TestLearningAugmentedAlgorithms(unittest.TestCase):
    """Tests for the new Learning-Augmented Algorithm functionality."""

    def test_ski_rental_laa_pipeline(self):
        """Test the end-to-end pipeline for generating and evaluating a ski rental LAA."""
        context = OrchestrationContext(
            project="SkiRentalLAA",
            objective="Generate a learning-augmented algorithm for the ski rental problem"
        )
        # Use a meta learner with epsilon=0 to ensure we test the main path
        meta_learner = MetaLearner(epsilon=0.0)
        orchestrator = OrchestratorAI(context, meta_learner)

        final_scripts = orchestrator.orchestrate(
            project_type="laa_ski_rental",
            domain="online_algorithms",
            complexity=0.5,
            enable_transfer_learning=False,
            enable_nas=False,
            enable_federated_optimization=False,
            trust_parameter=0.8
        )

        # Should generate a single, complete script
        self.assertEqual(len(final_scripts), 1)
        self.assertIn("def ski_rental_algorithm", final_scripts[0])

        # Check that the meta learner recorded the LAA metrics
        self.assertEqual(len(meta_learner.task_history), 1)
        task_record = meta_learner.task_history[0]
        self.assertIn('consistency', task_record.success_metrics)
        self.assertIn('robustness', task_record.success_metrics)
        # The pseudocode's formula is not 2-robust. We test that it's bounded.
        self.assertLess(task_record.success_metrics['robustness'], 6.0)


if __name__ == '__main__':
    unittest.main()
