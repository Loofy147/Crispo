import unittest
from orchestrator_core import (
    LayerParameters,
    GAOptimizer,
    RLAgent,
    MetaLearner,
    TaskMetadata
)

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


if __name__ == '__main__':
    unittest.main()
