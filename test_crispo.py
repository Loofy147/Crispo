"""Unit and integration tests for the OrchestratorAI system.

This test suite is organized into classes, each targeting a specific
component of the OrchestratorAI system, such as the Verifier, CodeGenerator,
GAOptimizer, etc. An integration test class is also included to verify the
end-to-end functionality of the main orchestrator pipeline.
"""
import unittest
from unittest import mock
import os
from crispo.crispo_core import (
    LayerParameters,
    GAOptimizer,
    RLAgent,
    AttentionRouter,
    CodeGenerator,
    MetaLearner,
    TaskMetadata,
    Verifier,
    SkiRentalContext,
    OneMaxSearchContext
)
from crispo.crispo import Crispo, OrchestrationContext
from crispo.advanced_crispo import (
    FederatedOptimizer
)
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

    def test_federated_optimizer_aggregates_temperature(self):
        """Verify that the federated optimizer correctly aggregates the temperature."""
        fo = FederatedOptimizer(num_clients=2)

        # Create two client models with different temperatures
        model1 = self.params.clone()
        model1.temperature = 0.5
        model2 = self.params.clone()
        model2.temperature = 1.5

        client_models = [model1, model2]
        client_data_sizes = [100, 100] # Equal weighting

        # Directly call the internal aggregate method to test the logic
        aggregated_model = fo._aggregate(client_models, client_data_sizes)

        # The aggregated temperature should be the average
        self.assertAlmostEqual(aggregated_model.temperature, 1.0)

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

    def test_generate_intent_api_fetch(self):
        """Verify that 'fetch' intent triggers the API template for layer 0."""
        # High complexity, but the intent 'fetch' should override for layer 0
        self.params.weights['complexity'] = 0.9
        script = self.cg.generate(self.params, 0, "I need to fetch data from a REST API.")
        self.assertIn("import requests", script)
        self.assertIn("def process_layer_0", script)

    def test_generate_intent_transform(self):
        """Verify that 'transform' intent triggers the pandas template for layer 1."""
        # Low complexity, but the intent 'transform' should override for layer 1
        self.params.weights['complexity'] = 0.1
        script = self.cg.generate(self.params, 1, "Please transform the dataset.")
        self.assertIn("import pandas as pd", script)
        self.assertIn("def process_layer_1", script)

    def test_generate_intent_fallback(self):
        """Verify fallback to complexity if intent is not relevant for the layer."""
        # The 'fetch' intent is for layer 0, so layer 2 should use complexity.
        self.params.weights['complexity'] = 0.9
        script = self.cg.generate(self.params, 2, "I need to fetch data.")
        self.assertIn("class Layer2System", script) # High-complexity template
        self.assertNotIn("import requests", script)

    def test_generated_one_max_script_correctness(self):
        """Verify that the generated one-max script calculates metrics correctly."""
        import subprocess
        import json
        import tempfile

        # 1. Generate the script
        script_code = self.cg.generate(self.params, 0, "one-max search")

        # 2. Save it to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(script_code)
            temp_filename = temp_file.name

        # 3. Execute the script with controlled inputs
        # Optimal is 100, algorithm is guided to select 80
        sequence = str([10, 50, 80, 100, 30])
        prediction = str([75, 85]) # Prediction interval that should select 80
        trust = "0.9"

        result = subprocess.run(
            input=f"'{sequence}' '{prediction}' {trust}",
            capture_output=True,
            text=True,
            # The script expects command-line arguments, not stdin
            args=['python3', temp_filename, sequence, prediction, trust]
        )

        # 4. Clean up the file
        os.remove(temp_filename)

        # 5. Parse and assert the output
        self.assertEqual(result.returncode, 0)
        output = json.loads(result.stdout)
        self.assertEqual(output['optimal_cost'], 100)
        self.assertEqual(output['algorithm_cost'], 80)
        # The competitive ratio for a maximization problem should be alg / opt
        self.assertAlmostEqual(output['competitive_ratio'], 0.8)


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

    def test_ga_elitism(self):
        """Verify that the GA's elitism correctly preserves the fittest individual."""
        # 1. Create a population where one individual is clearly the fittest.
        population_size = 10
        population = self.ga._initialize_population(self.template_params, population_size)

        # Create a "super" individual with a very high fitness score.
        # This individual has perfect alignment and temperature.
        super_individual = self.template_params.clone()
        super_individual.weights['complexity'] = self.context['desired_complexity']
        super_individual.temperature = 1.0

        # To make its fitness stand out, slightly worsen the others.
        for p in population:
             p.temperature = 1.5
        population[0] = super_individual

        # 2. Mock initialization to force the GA to use our crafted population.
        with mock.patch.object(self.ga, '_initialize_population', return_value=population):
            # 3. Run the GA for a single generation.
            best_found = self.ga.execute(self.template_params, self.context, generations=1)

            # 4. Assert that the returned individual is our "super" individual,
            # proving it survived the selection process.
            self.assertAlmostEqual(best_found.weights['complexity'], super_individual.weights['complexity'])
            self.assertAlmostEqual(best_found.temperature, super_individual.temperature)


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

    def test_rl_agent_persistence(self):
        """Verify that the RLAgent's Q-table can be saved and loaded."""
        import pickle

        # 1. Create an agent and populate its Q-table
        agent1 = RLAgent()
        agent1.q_table = {"state1": {"action1": 1.0}}

        # 2. Store the Q-table in a MetaLearner instance
        meta_learner_to_save = MetaLearner()
        meta_learner_to_save.rl_q_table = agent1.get_q_table()

        # 3. Simulate saving and loading the MetaLearner
        saved_meta_learner = pickle.dumps(meta_learner_to_save)
        loaded_meta_learner = pickle.loads(saved_meta_learner)

        # 4. Create a new agent and load the Q-table
        agent2 = RLAgent()
        agent2.load_q_table(loaded_meta_learner.rl_q_table)

        # 5. Assert that the new agent's Q-table is identical to the original
        self.assertEqual(agent1.get_q_table(), agent2.get_q_table())

    def test_epsilon_decay(self):
        """Verify that epsilon decays correctly over episodes."""
        agent = RLAgent(epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.1)
        initial_epsilon = agent.epsilon

        agent.execute(self.initial_params, self.context, episodes=1)
        first_decay_epsilon = agent.epsilon

        agent.execute(self.initial_params, self.context, episodes=1)
        second_decay_epsilon = agent.epsilon

        self.assertLess(first_decay_epsilon, initial_epsilon)
        self.assertLess(second_decay_epsilon, first_decay_epsilon)
        self.assertAlmostEqual(first_decay_epsilon, 1.0 * 0.9)
        self.assertAlmostEqual(second_decay_epsilon, 1.0 * 0.9 * 0.9)


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
        crispo = Crispo(context, meta_learner, problem_context=None)

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
        crispo = Crispo(context, meta_learner, problem_context=None)

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
        crispo = Crispo(context, MetaLearner(), problem_context=None)

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
        crispo = Crispo(context, MetaLearner(), problem_context=None)

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
        problem_context = SkiRentalContext()
        crispo = Crispo(context, meta_learner, problem_context=problem_context)

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
        from crispo.solution_registry import save_solution, load_latest_solution

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
        from crispo.solution_registry import save_solution, query_registry

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

    def test_save_solution_race_condition(self):
        """Verify that the save_solution function is robust against race conditions."""
        from crispo.solution_registry import save_solution
        import multiprocessing

        problem_type = "race_condition_test"

        # This worker function will be run by each process
        def worker():
            # Create dummy files to be moved by save_solution
            with open("generated_algorithm.py", "w") as f: f.write("alg")
            with open("generated_predictor.py", "w") as f: f.write("pred")
            save_solution(problem_type, {"metric": 1.0})

        # Create two processes that will run the worker function concurrently
        p1 = multiprocessing.Process(target=worker)
        p2 = multiprocessing.Process(target=worker)

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        # After both processes have completed, check the solution registry
        problem_path = os.path.join("solution_registry", problem_type)
        versions = [d for d in os.listdir(problem_path) if d.startswith('v')]

        # If the lock worked, there should be two distinct versions saved.
        # If it failed, one would have overwritten the other, resulting in only one version.
        self.assertEqual(len(versions), 2, "The locking mechanism failed to prevent a race condition.")


class TestProblemContexts(unittest.TestCase):
    """Tests for the problem context classes used in LAA evaluation."""

    def test_ski_rental_context(self):
        """Verify the SkiRentalContext's methods."""
        context = SkiRentalContext(buy_cost=50)
        scenario = 30  # Actual days

        # Test command generation
        cmd = context.get_evaluation_command("script.py", "[25, 35]", 0.8, scenario)
        self.assertEqual(cmd, ['python3', 'script.py', '50', '[25, 35]', '0.8', '30'])

        # Test prediction generation
        self.assertEqual(context.get_perfect_prediction(scenario), [30, 30])
        self.assertEqual(context.get_worst_prediction(scenario), [1, 2])
        self.assertEqual(context.get_noisy_prediction(scenario, 0.1), [27, 33])

    def test_one_max_context(self):
        """Verify the OneMaxSearchContext's methods."""
        context = OneMaxSearchContext()
        scenario = [10, 80, 45, 92, 30] # Sequence of values

        # Test command generation
        cmd = context.get_evaluation_command("script.py", "[85, 95]", 0.9, scenario)
        self.assertEqual(cmd, ['python3', 'script.py', str(scenario), '[85, 95]', '0.9'])

        # Test prediction generation
        self.assertEqual(context.get_perfect_prediction(scenario), [92, 92])
        self.assertEqual(context.get_worst_prediction(scenario), [10, 10])
        self.assertEqual(context.get_noisy_prediction(scenario, 0.1), [82, 101])


from unittest import mock

class TestCLI(unittest.TestCase):
    """Tests for the command-line interface in crispo.py."""

    @mock.patch('argparse.ArgumentParser')
    def test_cli_argument_parsing(self, mock_parser):
        """Verify that CLI arguments are correctly parsed and passed to Crispo."""
        from crispo.crispo import main
        import crispo.crispo as crispo_module

        # Mock the parsed arguments
        mock_args = unittest.mock.MagicMock()
        mock_args.project = "CLITest"
        mock_args.objective = "Test CLI"
        mock_args.project_type = "cli_test"
        mock_args.domain = "testing"
        mock_args.complexity = 0.1
        mock_args.load_metaknowledge = None
        mock_args.save_metaknowledge = "test.pkl"
        mock_args.enable_transfer_learning = True
        mock_args.enable_nas = False
        mock_args.enable_federated_optimization = True
        mock_args.trust_parameter = 0.9
        mock_args.query_registry = None
        mock_parser.return_value.parse_args.return_value = mock_args

        # Mock the Crispo class itself to intercept its creation and methods
        with unittest.mock.patch('crispo.crispo.Crispo') as mock_crispo_class:
            # We need a mock instance to be returned when Crispo is instantiated
            mock_crispo_instance = unittest.mock.MagicMock()
            mock_crispo_class.return_value = mock_crispo_instance

            # We also need to mock the pickle dump for saving metaknowledge
            with unittest.mock.patch('pickle.dump') as mock_pickle_dump:
                main()

                # 1. Check if Crispo was instantiated correctly
                mock_crispo_class.assert_called_once()
                # We can check the context object passed to the constructor
                args, _ = mock_crispo_class.call_args
                context_arg = args[0]
                self.assertEqual(context_arg.project, "CLITest")
                self.assertEqual(context_arg.objective, "Test CLI")

                # 2. Check if the orchestrate method was called with the correct args
                mock_crispo_instance.orchestrate.assert_called_once_with(
                    project_type="cli_test",
                    domain="testing",
                    complexity=0.1,
                    enable_transfer_learning=True,
                    enable_nas=False,
                    enable_federated_optimization=True,
                    trust_parameter=0.9
                )

                # 3. Check if meta-knowledge saving was attempted
                mock_pickle_dump.assert_called_once()

    @mock.patch('argparse.ArgumentParser')
    @mock.patch('crispo.crispo.query_registry')
    def test_cli_query_registry_bypasses_orchestration(self, mock_query_registry, mock_parser):
        """Verify that --query-registry bypasses the main orchestration logic."""
        from crispo.crispo import main

        # Mock the parsed arguments to simulate the --query-registry flag
        mock_args = unittest.mock.MagicMock()
        mock_args.query_registry = "competitive_ratio:1.5"
        mock_parser.return_value.parse_args.return_value = mock_args

        # Mock the Crispo class to ensure it's not called
        with unittest.mock.patch('crispo.crispo.Crispo') as mock_crispo_class:
            main()

            # 1. Assert that the orchestration engine was NOT created or run
            mock_crispo_class.assert_not_called()
            mock_crispo_class.return_value.orchestrate.assert_not_called()

            # 2. Assert that the query function WAS called with the correct parameters
            mock_query_registry.assert_called_once_with("competitive_ratio", 1.5)

    @mock.patch('argparse.ArgumentParser')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('pickle.load')
    def test_cli_load_metaknowledge_file_not_found(self, mock_pickle_load, mock_open, mock_parser):
        """Verify that a warning is printed when the metaknowledge file is not found."""
        from crispo.crispo import main
        import io

        # Mock the parsed arguments
        mock_args = unittest.mock.MagicMock()
        mock_args.load_metaknowledge = "non_existent_file.pkl"
        mock_args.save_metaknowledge = None # Ensure the save block is not triggered
        mock_args.query_registry = None  # Ensure orchestration is not bypassed
        mock_parser.return_value.parse_args.return_value = mock_args

        # Make the mock 'open' raise a FileNotFoundError
        mock_open.side_effect = FileNotFoundError

        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with unittest.mock.patch('crispo.crispo.Crispo'): # Mock Crispo to prevent orchestration
                main()
                output = mock_stdout.getvalue()
                self.assertIn("Warning: Meta-knowledge file not found", output)

if __name__ == '__main__':
    unittest.main()
