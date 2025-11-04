import unittest
from orchestrator_core import (
    LayerParameters,
    GAOptimizer,
    RLAgent,
    AttentionRouter,
    CodeGenerator,
    MetaLearner,
    TaskMetadata
)
from orchestrator import OrchestratorAI, OrchestrationContext
from advanced_orchestrator import (
    TransferLearningEngine,
    NeuralArchitectureSearch,
    FederatedOptimizer
)

class TestAdvancedFeatures(unittest.TestCase):

    def setUp(self):
        self.params = LayerParameters(
            layer_id=0,
            weights={'complexity': 1.0, 'execution': 1.0},
            biases={'logging': 0.0},
            temperature=1.0
        )

    def test_transfer_learning_engine(self):
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
        nas = NeuralArchitectureSearch(search_space={'layer_type': ['dense']})
        architecture = nas.search(num_layers=3)
        self.assertEqual(len(architecture), 3)
        self.assertEqual(architecture[0]['layer_type'], ['dense'])

    def test_federated_optimizer(self):
        fo = FederatedOptimizer(num_clients=5)
        optimized_params = fo.optimize(self.params)
        self.assertEqual(optimized_params, self.params)

class TestCodeGenerator(unittest.TestCase):

    def setUp(self):
        self.cg = CodeGenerator()
        self.params = LayerParameters(
            layer_id=0,
            weights={'complexity': 0.9, 'execution': 1.0},
            biases={'logging': 0.1},
            temperature=1.2
        )

    def test_generate_high_complexity(self):
        script = self.cg.generate(self.params, 0)
        self.assertIn("class Layer0System", script)
        self.assertIn("import numpy as np", script)

    def test_generate_medium_complexity(self):
        self.params.weights['complexity'] = 0.6
        script = self.cg.generate(self.params, 1)
        self.assertIn("import pandas as pd", script)
        self.assertIn("def process_layer_1", script)

    def test_generate_low_complexity(self):
        self.params.weights['complexity'] = 0.1
        script = self.cg.generate(self.params, 2)
        self.assertNotIn("class", script)
        self.assertIn("def process_layer_2", script)

class TestAttentionRouter(unittest.TestCase):

    def setUp(self):
        self.ar = AttentionRouter(embed_dim=4, num_heads=2)
        self.query = [1, 0, 0, 0]
        self.keys = [[1, 0, 0, 0], [0, 1, 0, 0]]
        self.values = [[0, 0, 1, 0], [0, 0, 0, 1]]

    def test_execute(self):
        result = self.ar.execute(self.query, self.keys, self.values)
        self.assertIn('attended_output', result)
        self.assertIn('attention_weights', result)
        self.assertEqual(len(result['attended_output']), 4)
        self.assertAlmostEqual(sum(result['attention_weights']), 1.0)

class TestGAOptimizer(unittest.TestCase):

    def setUp(self):
        self.ga = GAOptimizer(population_size=10, mutation_rate=0.5)
        self.template_params = LayerParameters(
            layer_id=0,
            weights={'complexity': 1.0, 'execution': 1.0},
            biases={'logging': 0.0},
            temperature=1.0
        )
        self.context = {'desired_complexity': 0.8}

    def test_initialization(self):
        self.assertEqual(self.ga.population_size, 10)
        self.assertEqual(self.ga.mutation_rate, 0.5)

    def test_execute(self):
        result = self.ga.execute(self.template_params, self.context, generations=5)
        self.assertIsInstance(result, LayerParameters)
        self.assertIn('complexity', result.weights)

class TestRLAgent(unittest.TestCase):

    def setUp(self):
        self.rl = RLAgent(epsilon=0.1)
        self.initial_params = LayerParameters(
            layer_id=0,
            weights={'complexity': 1.0, 'execution': 1.0},
            biases={'logging': 0.0},
            temperature=1.0
        )
        self.context = {'complexity': 0.8}

    def test_initialization(self):
        self.assertEqual(self.rl.epsilon, 0.1)

    def test_execute(self):
        result = self.rl.execute(self.initial_params, self.context, episodes=3)
        self.assertIsInstance(result, LayerParameters)
        self.assertNotEqual(result.temperature, self.initial_params.temperature)

class TestMetaLearner(unittest.TestCase):

    def setUp(self):
        self.ml = MetaLearner()
        self.task1 = TaskMetadata(
            task_id="1",
            project_type="data_pipeline",
            complexity_level=0.8,
            domain="data_engineering",
            success_metrics={'overall_quality': 0.9},
            optimal_config={'ga_generations': 25, 'rl_episodes': 12},
            timestamp=""
        )
        self.task2 = TaskMetadata(
            task_id="2",
            project_type="web_scraper",
            complexity_level=0.4,
            domain="web_services",
            success_metrics={'overall_quality': 0.7},
            optimal_config={'ga_generations': 5, 'rl_episodes': 3},
            timestamp=""
        )

    def test_record_and_get_strategy(self):
        self.ml.record_task(self.task1)
        self.ml.record_task(self.task2)

        strategy_dp = self.ml.get_optimal_strategy("data_pipeline", 0.8)
        self.assertEqual(strategy_dp['ga_generations'], 20)
        self.assertEqual(strategy_dp['rl_episodes'], 9)

        strategy_ws = self.ml.get_optimal_strategy("web_scraper", 0.4)
        self.assertEqual(strategy_ws['ga_generations'], 5)
        self.assertEqual(strategy_ws['rl_episodes'], 3)

        strategy_unknown = self.ml.get_optimal_strategy("unknown_type", 0.5)
        self.assertEqual(strategy_unknown['ga_generations'], 5)
        self.assertEqual(strategy_unknown['rl_episodes'], 2)

class TestOrchestratorIntegration(unittest.TestCase):

    def test_full_pipeline(self):
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
            enable_federated_optimization=False
        )

        self.assertEqual(len(final_scripts), 3)
        self.assertIsInstance(final_scripts[0], str)
        self.assertIn("# Layer 0", final_scripts[0])
        self.assertIn("# Layer 1", final_scripts[1])
        self.assertIn("# Layer 2", final_scripts[2])

    def test_full_pipeline_with_advanced_features(self):
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
            enable_federated_optimization=True
        )

        self.assertEqual(len(final_scripts), 3)
        self.assertIsInstance(final_scripts[0], str)


if __name__ == '__main__':
    unittest.main()
