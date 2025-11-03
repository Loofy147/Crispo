import unittest
import math
from orchestrator import GAOptimizer, RLAgent, AttentionRouter, OrchestratorAI, OrchestrationContext

class TestGAOptimizer(unittest.TestCase):

    def setUp(self):
        """Set up a common GAOptimizer instance and parameters for tests."""
        self.optimizer = GAOptimizer(population_size=50, mutation_rate=0.1)
        self.params_template = {
            'weights': {'complexity': 1.0, 'execution': 1.0},
            'biases': {'logging': 0.0},
            'temperature': 1.0
        }

    def test_initialize_population_size(self):
        """Verify that the population is initialized with the correct size."""
        population = self.optimizer._initialize_population(self.params_template)
        self.assertEqual(len(population), 50)

    def test_evaluate_fitness_returns_float(self):
        """Ensure the fitness function returns a float within an expected range."""
        individual = self.optimizer._initialize_population(self.params_template)[0]
        fitness_score = self.optimizer._evaluate_fitness(individual, {})
        self.assertIsInstance(fitness_score, float)
        self.assertGreaterEqual(fitness_score, 0.0)
        self.assertLessEqual(fitness_score, 3.0)

    def test_crossover_returns_two_children(self):
        """Verify that the crossover function returns two offspring."""
        population = self.optimizer._initialize_population(self.params_template)
        parent1, parent2 = population[0], population[1]
        child1, child2 = self.optimizer._crossover(parent1, parent2)
        self.assertIn('weights', child1)
        self.assertIn('temperature', child2)
        self.assertEqual(len(child1['weights']), len(parent1['weights']))

    def test_mutate_applies_mutation(self):
        """Check that the mutate function modifies parameters within bounds."""
        individual = self.optimizer._initialize_population(self.params_template)[0]
        mutated_individual = self.optimizer._mutate(individual)
        self.assertGreaterEqual(mutated_individual['temperature'], 0.5)
        self.assertLessEqual(mutated_individual['temperature'], 2.0)
        for weight in mutated_individual['weights'].values():
            self.assertGreaterEqual(weight, 0.1)
            self.assertLessEqual(weight, 2.0)

class TestRLAgent(unittest.TestCase):

    def setUp(self):
        """Set up a common RLAgent instance and parameters for tests."""
        self.agent = RLAgent()
        self.params = {
            'weights': {'complexity': 1.0, 'execution': 1.0},
            'temperature': 1.0
        }

    def test_encode_state_returns_string(self):
        """Verify that the state encoding function returns a non-empty string."""
        state = self.agent._encode_state(self.params, {'complexity': 0.8})
        self.assertIsInstance(state, str)
        self.assertTrue(len(state) > 0)

    def test_apply_action_modifies_params(self):
        """Ensure that applying an action correctly modifies the parameters."""
        action = {'weight_delta': 0.1, 'temp_delta': -0.1}
        new_params = self.agent._apply_action(self.params, action)
        self.assertAlmostEqual(new_params['weights']['complexity'], 1.1)
        self.assertAlmostEqual(new_params['temperature'], 0.9)

    def test_calculate_reward_returns_float(self):
        """Check that the reward calculation returns a float."""
        reward = self.agent._calculate_reward(self.params, {})
        self.assertIsInstance(reward, float)

class TestAttentionRouter(unittest.TestCase):

    def setUp(self):
        """Set up a common AttentionRouter instance."""
        self.router = AttentionRouter()

    def test_calculate_entropy_returns_non_negative(self):
        """Ensure that entropy calculation is always non-negative."""
        weights = [0.1, 0.2, 0.7]
        entropy = self.router._calculate_entropy(weights)
        self.assertGreaterEqual(entropy, 0)

    def test_execute_attention_weights_sum_to_one(self):
        """Verify that attention weights sum to approximately 1.0."""
        query = [0.1] * 16
        keys = [[0.2] * 16, [0.3] * 16]
        values = keys
        result = self.router.execute(query, keys, values)
        self.assertAlmostEqual(sum(result['attention_weights']), 1.0, places=6)

class TestOrchestratorAI(unittest.TestCase):

    def setUp(self):
        """Set up a common OrchestratorAI instance for integration tests."""
        context = OrchestrationContext(project="TestProject", objective="TestObjective")
        self.orchestrator = OrchestratorAI(context)

    def test_orchestrate_runs_and_returns_results(self):
        """Verify that the main orchestration process runs without errors."""
        results = self.orchestrator.orchestrate(generations=1, episodes=1)
        self.assertIn('ga_optimization', results)
        self.assertIn('rl_optimization', results)
        self.assertIn('attention_routing', results)
        self.assertIn('code_generation', results)

    def test_generate_code_produces_correct_layers(self):
        """Check that the code generation method produces the correct number of layers."""
        params = {'weights': {'complexity': 0.8}, 'temperature': 1.0}
        attention = {'attention_weights': [0.5, 0.5]}
        code_result = self.orchestrator._generate_code(params, attention)
        self.assertEqual(code_result['layers_generated'], 3)
        self.assertEqual(len(code_result['generated_scripts']), 3)

if __name__ == '__main__':
    unittest.main()
