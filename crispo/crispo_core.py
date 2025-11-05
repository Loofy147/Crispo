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

@dataclass
class ScriptToken:
    """Represents a token in the script generation vocabulary.

    Attributes:
        value: The string value of the token.
        weight: The weight associated with the token, used in generation.
    """
    value: str
    weight: float = 1.0

@dataclass
class LayerParameters:
    """Defines the parameters for generating a single script layer.

    Attributes:
        layer_id: The unique identifier for the layer.
        weights: A dictionary of weights influencing generation.
        biases: A dictionary of biases influencing generation.
        temperature: A value controlling the randomness of the generation process.
    """
    layer_id: int
    weights: Dict[str, float]
    biases: Dict[str, float]
    temperature: float = 1.0

    def clone(self) -> 'LayerParameters':
        """Creates a deep copy of the LayerParameters instance.

        This is primarily used to create new individuals in the genetic algorithm
        without modifying the original parameters.

        Returns:
            A new LayerParameters instance with the same attribute values.
        """
        return LayerParameters(
            layer_id=self.layer_id,
            weights=self.weights.copy(),
            biases=self.biases.copy(),
            temperature=self.temperature
        )

# ============================================================================
# PRODUCTION UNIT: GENETIC ALGORITHM OPTIMIZER
# ============================================================================

class GAOptimizer:
    """Evolves layer configurations using a genetic algorithm.

    This class implements a standard genetic algorithm to search for an optimal
    set of LayerParameters. It includes selection, crossover, and mutation
    operators to evolve a population of parameter sets over a number of
    generations. An elitism mechanism is used to ensure the best individual
    from one generation is carried over to the next.
    """

    def __init__(self, base_population_size: int = 20, mutation_rate: float = 0.15, crossover_rate: float = 0.7):
        """Initializes the GAOptimizer.

        Args:
            base_population_size (int): The base number of individuals for a
                low-complexity problem.
            mutation_rate (float): The probability of an individual's genes
                (parameters) being randomly altered.
            crossover_rate (float): The probability that two parents will
                exchange genetic material to create offspring.
        """
        self.base_population_size = base_population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def execute(self, template_params: LayerParameters, context: Dict, generations: int = 10) -> LayerParameters:
        """Runs the genetic algorithm to find the best LayerParameters.

        The algorithm iteratively refines a population of LayerParameters
        instances through selection, crossover, and mutation.

        Args:
            template_params (LayerParameters): An initial LayerParameters
                instance to seed the population.
            context (Dict): A dictionary providing context (e.g.,
                'desired_complexity') for the fitness evaluation.
            generations (int): The number of generations to run the evolution.

        Returns:
            LayerParameters: The best LayerParameters instance found after all
                generations.
        """
        complexity = context.get('desired_complexity', 0.5)
        population_size = int(self.base_population_size * (1 + complexity))
        population = self._initialize_population(template_params, population_size)

        for gen in range(generations):
            fitness_scores = [self._evaluate_fitness(ind, context) for ind in population]

            print(f"  [GA] Gen {gen + 1}: Best Fitness={max(fitness_scores):.3f}, Avg Fitness={sum(fitness_scores)/len(fitness_scores):.3f}")

            parents = self._tournament_selection(population, fitness_scores)

            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self._crossover(parents[i], parents[i+1])
                offspring.extend([child1, child2])

            offspring = [self._mutate(child) for child in offspring]

            # New elitism strategy: Combine parents and offspring, then select the best
            combined_population = population + offspring
            combined_fitness = [self._evaluate_fitness(ind, context) for ind in combined_population]

            # Sort the combined population by fitness and select the top individuals
            sorted_combined = [x for _, x in sorted(zip(combined_fitness, combined_population), key=lambda pair: pair[0], reverse=True)]
            population = sorted_combined[:population_size]


        final_fitness = [self._evaluate_fitness(ind, context) for ind in population]
        best_idx = final_fitness.index(max(final_fitness))
        return population[best_idx]

    def _initialize_population(self, template: LayerParameters, population_size: int) -> List[LayerParameters]:
        """Creates an initial population of LayerParameters.

        Args:
            template: The LayerParameters instance to use as a template.
            population_size: The number of individuals to create.

        Returns:
            A list of new LayerParameters instances with randomized attributes.
        """
        population = []
        for _ in range(population_size):
            individual = template.clone()
            for key in individual.weights:
                individual.weights[key] *= random.uniform(0.5, 1.5)
            for key in individual.biases:
                individual.biases[key] += random.uniform(-0.2, 0.2)
            individual.temperature = random.uniform(0.7, 1.3)
            population.append(individual)
        return population

    def _evaluate_fitness(self, params: LayerParameters, context: Dict) -> float:
        """Calculates the fitness of a single LayerParameters instance.

        The fitness function rewards parameter sets that are well-balanced and
        align with the desired complexity from the context.

        Args:
            params: The LayerParameters instance to evaluate.
            context: A dictionary containing contextual information, like
                'desired_complexity'.

        Returns:
            A float representing the fitness score, where higher is better.
        """
        fitness = 0.0
        # Reward balanced weights
        avg_weight = sum(params.weights.values()) / len(params.weights)
        weight_variance = sum((w - avg_weight) ** 2 for w in params.weights.values()) / len(params.weights)
        fitness += max(0, 1.0 - weight_variance)
        # Reward moderate temperature
        fitness += max(0, 1.0 - abs(params.temperature - 1.0))
        # Reward context alignment
        if 'desired_complexity' in context:
            alignment = 1.0 - abs(params.weights.get('complexity', 1.0) - context['desired_complexity'])
            fitness += alignment
        return max(0.0, fitness)

    def _tournament_selection(self, population: List[LayerParameters], fitness: List[float]) -> List[LayerParameters]:
        """Selects parents from the population using tournament selection.

        Args:
            population: The current population of LayerParameters instances.
            fitness: A list of fitness scores corresponding to the population.

        Returns:
            A list of selected parent LayerParameters instances.
        """
        parents = []
        tournament_size = 3
        if len(population) < tournament_size:
            return population
        for _ in range(len(population) // 2):
            indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness[i] for i in indices]
            winner_idx = indices[tournament_fitness.index(max(tournament_fitness))]
            parents.append(population[winner_idx])
        return parents

    def _crossover(self, parent1: LayerParameters, parent2: LayerParameters) -> Tuple[LayerParameters, LayerParameters]:
        """Performs single-point crossover between two parents.

        Args:
            parent1: The first parent LayerParameters instance.
            parent2: The second parent LayerParameters instance.

        Returns:
            A tuple containing two new child LayerParameters instances.
        """
        if random.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()

        child1, child2 = parent1.clone(), parent2.clone()
        weight_keys = list(parent1.weights.keys())
        crossover_point = random.randint(0, len(weight_keys))
        for i, key in enumerate(weight_keys):
            if i < crossover_point:
                child1.weights[key] = parent2.weights[key]
                child2.weights[key] = parent1.weights[key]
        if random.random() < 0.5:
            child1.temperature, child2.temperature = child2.temperature, child1.temperature
        return child1, child2

    def _mutate(self, params: LayerParameters) -> LayerParameters:
        """Randomly mutates the attributes of a LayerParameters instance.

        Args:
            params: The LayerParameters instance to mutate.

        Returns:
            A new, mutated LayerParameters instance.
        """
        mutated = params.clone()
        for key in mutated.weights:
            if random.random() < self.mutation_rate:
                mutated.weights[key] *= random.uniform(0.8, 1.2)
        for key in mutated.biases:
            if random.random() < self.mutation_rate:
                mutated.biases[key] += random.uniform(-0.1, 0.1)
        if random.random() < self.mutation_rate:
            mutated.temperature += random.uniform(-0.2, 0.2)
        return mutated

# ============================================================================
# PRODUCTION UNIT: REINFORCEMENT LEARNING AGENT
# ============================================================================

class RLAgent:
    """Uses a Q-learning algorithm to fine-tune LayerParameters.

    This agent interacts with an environment defined by the LayerParameters,
    taking actions to modify them and receiving rewards based on their quality.
    It learns a policy to maximize these rewards over time using a Q-table to
    store state-action values.
    """

    def __init__(self, learning_rate: float = 0.1, discount: float = 0.95, epsilon: float = 1.0, epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        """Initializes the RLAgent.

        Args:
            learning_rate (float): The rate at which the agent learns from new
                information (alpha in the Bellman equation).
            discount (float): The discount factor for future rewards (gamma).
            epsilon (float): The starting exploration rate.
            epsilon_decay (float): The rate at which epsilon decays after each
                episode.
            min_epsilon (float): The minimum value epsilon can decay to.
        """
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table: Dict[str, Dict[str, float]] = {}

    def load_q_table(self, q_table: Dict[str, Dict[str, float]]):
        """Loads a pre-trained Q-table into the agent."""
        self.q_table = q_table

    def get_q_table(self) -> Dict[str, Dict[str, float]]:
        """Returns the agent's current Q-table."""
        return self.q_table

    def execute(self, initial_params: LayerParameters, context: Dict, episodes: int = 5) -> LayerParameters:
        """Runs the RL fine-tuning process.

        The agent iteratively refines a set of LayerParameters by taking actions,
        observing rewards, and updating its Q-table.

        Args:
            initial_params (LayerParameters): The starting LayerParameters to
                be tuned.
            context (Dict): A dictionary providing context for state encoding.
            episodes (int): The number of episodes to run the training.

        Returns:
            LayerParameters: The best LayerParameters instance found during
                the episodes.
        """
        best_params = initial_params.clone()
        best_reward = float('-inf')

        for episode in range(episodes):
            params = initial_params.clone()
            episode_reward = 0.0

            for step in range(3): # 3 steps per episode
                state = self._encode_state(params, context)
                action = self._select_action(state)
                params = self._apply_action(params, action)
                reward = self._calculate_reward(params)
                episode_reward += reward
                next_state = self._encode_state(params, context)
                self._update_q_value(state, action, reward, next_state)

            print(f"  [RL] Episode {episode + 1}: Reward={episode_reward:.3f}, Epsilon={self.epsilon:.3f}")

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_params = params.clone()

            # Decay epsilon after each episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return best_params

    def _encode_state(self, params: LayerParameters, context: Dict) -> str:
        """Encodes the current state into a string for Q-table lookup.

        Args:
            params: The current LayerParameters.
            context: A dictionary containing additional context.

        Returns:
            A string representation of the state.
        """
        state_vec = [
            params.temperature,
            context.get('complexity', 0.5)
        ]

        # Add all weights and biases to the state vector
        for key in sorted(params.weights.keys()):
            state_vec.append(params.weights[key])
        for key in sorted(params.biases.keys()):
            state_vec.append(params.biases[key])

        return str(tuple(round(v, 1) for v in state_vec))

    def _select_action(self, state: str) -> Dict:
        """Selects an action using an epsilon-greedy policy.

        With probability epsilon, it explores a random action. Otherwise, it
        exploits the best-known action for the given state.

        Args:
            state: The current state string.

        Returns:
            A dictionary representing the action to take.
        """
        if random.random() < self.epsilon or state not in self.q_table or not self.q_table[state]:
            return {
                'weight_delta': random.uniform(-0.1, 0.1),
                'bias_delta': random.uniform(-0.05, 0.05),
                'temp_delta': random.uniform(-0.1, 0.1)
            }
        best_action_str = max(self.q_table[state], key=self.q_table[state].get)
        return json.loads(best_action_str)

    def _apply_action(self, params: LayerParameters, action: Dict) -> LayerParameters:
        """Applies an action to a LayerParameters instance.

        Args:
            params: The LayerParameters to modify.
            action: The action dictionary defining the modifications.

        Returns:
            A new LayerParameters instance with the action applied.
        """
        new_params = params.clone()
        for key in new_params.weights:
            new_params.weights[key] += action['weight_delta']
        for key in new_params.biases:
            new_params.biases[key] += action['bias_delta']
        new_params.temperature += action['temp_delta']
        return new_params

    def _calculate_reward(self, params: LayerParameters) -> float:
        """Calculates the reward for a given set of LayerParameters.

        The reward function encourages balanced weights and a moderate
        temperature value.

        Args:
            params: The LayerParameters to evaluate.

        Returns:
            A float representing the reward.
        """
        reward = 0.0
        # Reward balanced weights
        weights = params.weights
        if weights:
            avg = sum(weights.values()) / len(weights)
            variance = sum((w - avg) ** 2 for w in weights.values()) / len(weights)
            reward += max(0, 1.0 - variance)
        # Reward moderate temperature
        reward += max(0, 1.0 - abs(params.temperature - 1.0))
        return reward

    def _update_q_value(self, state: str, action: Dict, reward: float, next_state: str):
        """Updates the Q-table using the Bellman equation.

        Args:
            state: The state in which the action was taken.
            action: The action that was taken.
            reward: The reward received for the action.
            next_state: The resulting state after the action.
        """
        action_str = json.dumps(action, sort_keys=True)
        if state not in self.q_table:
            self.q_table[state] = {}
        if action_str not in self.q_table[state]:
            self.q_table[state][action_str] = 0.0

        max_next_q = 0.0
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())

        current_q = self.q_table[state][action_str]
        new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q)
        self.q_table[state][action_str] = new_q

# ============================================================================
# PRODUCTION UNIT: ATTENTION ROUTER
# ============================================================================

class AttentionRouter:
    """Implements a multi-head attention mechanism for inter-layer communication.

    This class allows different layers in the generated pipeline to share
    information, enabling more cohesive and context-aware code generation. It
    uses a simplified dot-product attention to weigh the importance of
    different value vectors based on a query vector.
    """

    def __init__(self, embed_dim: int = 16, num_heads: int = 4):
        """Initializes the AttentionRouter.

        Args:
            embed_dim (int): The dimensionality of the embedding space.
            num_heads (int): The number of attention heads to use.
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def execute(self, query: List[float], keys: List[List[float]], values: List[List[float]]) -> Dict:
        """Executes the attention mechanism.

        It calculates attention scores, applies a softmax to get weights, and
        then computes a weighted sum of the values.

        Args:
            query (List[float]): The query vector, representing the current
                context.
            keys (List[List[float]]): A list of key vectors, corresponding to
                the items to be attended to.
            values (List[List[float]]): A list of value vectors, containing the
                information to be retrieved.

        Returns:
            Dict: A dictionary containing the 'attended_output' (a list of
                floats) and the 'attention_weights' (a list of floats).
        """
        if not keys or not values:
            return {'attended_output': query, 'attention_weights': []}

        scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(self.embed_dim) for key in keys]

        exp_scores = [math.exp(s) for s in scores]
        sum_exp = sum(exp_scores)
        attention_weights = [s / sum_exp for s in exp_scores]

        output = [0.0] * self.embed_dim
        for weight, value in zip(attention_weights, values):
            for i in range(min(len(output), len(value))):
                output[i] += weight * value[i]

        return {
            'attended_output': output,
            'attention_weights': attention_weights
        }

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

    def generate(self, params: LayerParameters, layer_id: int, objective: str, trust_parameter: float = 0.8) -> str:
        """Selects and generates a script for a single layer.

        This method uses keyword matching on the user's objective to select
        the most logically appropriate template for each layer of the pipeline.
        If a clear intent cannot be determined, it falls back to a
        complexity-based selection mechanism.

        Args:
            params (LayerParameters): The LayerParameters for the current
                layer, which influence the generated code's specifics.
            layer_id (int): The ID of the current layer (e.g., 0, 1, 2).
            objective (str): The user's high-level objective string, used for
                intent detection.
            trust_parameter (float): The trust parameter (lambda) for
                learning-augmented algorithms.

        Returns:
            str: A string containing the generated Python script.
        """
        objective = objective.lower()

        # Check for LAA-specific objectives first
        if "ski rental" in objective:
            return self._generate_ski_rental_laa_template(params, trust_parameter)
        if "one-max search" in objective:
            return self._generate_one_max_laa_template(params, trust_parameter)

        # Layer 0: Prioritize fetching data
        if layer_id == 0 and any(kw in objective for kw in ['fetch', 'get', 'api', 'request']):
            return self._generate_api_template(params, layer_id)

        # Layer 1: Prioritize transformation
        if layer_id == 1 and any(kw in objective for kw in ['process', 'transform', 'clean', 'pandas']):
            return self._generate_transform_template(params, layer_id)

        # Layer 2: Prioritize analysis
        if layer_id == 2 and any(kw in objective for kw in ['analyze', 'numpy', 'compute', 'calculate']):
            return self._generate_high_complexity_template(params, layer_id)

        # Fallback to complexity-based logic if intent is not clear for the layer
        complexity = params.weights.get('complexity', 1.0)
        if complexity > 0.8:
            return self._generate_high_complexity_template(params, layer_id)
        if complexity > 0.5:
            return self._generate_transform_template(params, layer_id)
        if complexity > 0.2:
            return self._generate_api_template(params, layer_id)

        return self._generate_simple_template(params, layer_id)

    def _generate_ski_rental_laa_template(self, params: LayerParameters, trust_parameter: float) -> str:
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

    def _generate_one_max_laa_template(self, params: LayerParameters, trust_parameter: float) -> str:
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

    def _generate_high_complexity_template(self, params: LayerParameters, layer_id: int) -> str:
        """Generates a script for high-complexity data processing using NumPy.

        Args:
            params: The LayerParameters for the current layer.
            layer_id: The ID of the current layer.

        Returns:
            A string containing the generated Python script.
        """
        return f'''# Layer {layer_id}: High-Complexity Data Processing
import numpy as np
import json

class Layer{layer_id}System:
    def __init__(self):
        self.weights = np.array({list(params.weights.values())})
        self.biases = np.array({list(params.biases.values())})
        self.temp = {params.temperature:.2f}

    def process(self, input_context):
        data = np.array(input_context.get('data', []))
        # Apply a non-linear transformation
        transformed_data = np.tanh(np.dot(data, self.weights.T) + self.biases)
        output_context = {{'data': transformed_data.tolist()}}
        return output_context

if __name__ == '__main__':
    system = Layer{layer_id}System()
    # Example usage with mock data
    mock_input_context = {{'data': np.random.rand(1, {len(params.weights)}).tolist()}}
    result = system.process(mock_input_context)
    print(json.dumps(result))
'''

    def _generate_transform_template(self, params: LayerParameters, layer_id: int) -> str:
        """Generates a script for data transformation using pandas.

        Args:
            params: The LayerParameters for the current layer.
            layer_id: The ID of the current layer.

        Returns:
            A string containing the generated Python script.
        """
        return f'''# Layer {layer_id}: Data Transformation
import pandas as pd
import json

def process_layer_{layer_id}(input_context):
    data = input_context.get('data', {{}})
    df = pd.DataFrame(data)
    # Perform a simple data transformation
    if not df.empty:
        df['new_col'] = df.iloc[:, 0] * {params.weights.get('execution', 1.0):.2f}
    output_context = {{'data': df.to_dict('records')}}
    return output_context

if __name__ == '__main__':
    # Example usage with mock data
    mock_input_context = {{'data': {{'col1': [1, 2, 3], 'col2': [4, 5, 6]}}}}
    result = process_layer_{layer_id}(mock_input_context)
    print(json.dumps(result))
'''

    def _generate_api_template(self, params: LayerParameters, layer_id: int) -> str:
        """Generates a script for interacting with an API using requests.

        Args:
            params: The LayerParameters for the current layer.
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
        response = requests.get(api_endpoint, timeout=API_REQUEST_TIMEOUT)
        response.raise_for_status()
        output_context = {{'data': response.json()}}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {{e}}")
        output_context = {{'data': [], 'error': str(e)}}
    return output_context

if __name__ == '__main__':
    # Example usage with a mock API endpoint
    mock_input_context = {{'api_endpoint': 'https://jsonplaceholder.typicode.com/todos/1'}}
    result = process_layer_{layer_id}(mock_input_context)
    print(json.dumps(result))
'''

    def _generate_simple_template(self, params: LayerParameters, layer_id: int) -> str:
        """Generates a script for simple, low-complexity processing.

        Args:
            params: The LayerParameters for the current layer.
            layer_id: The ID of the current layer.

        Returns:
            A string containing the generated Python script.
        """
        return f'''# Layer {layer_id}: Simple Processing
import json

def process_layer_{layer_id}(input_context):
    data = input_context.get('data', 0)
    weight = {params.weights.get("execution", 1.0):.2f}
    bias = {params.biases.get("execution", 0.0):.2f}
    result = data * weight + bias
    output_context = {{'data': result}}
    return output_context

if __name__ == '__main__':
    # Example usage with mock data
    mock_input_context = {{'data': 10}}
    result = process_layer_{layer_id}(mock_input_context)
    print(json.dumps(result))
'''

# ============================================================================
# ADVANCED FEATURE: META-LEARNING
# ============================================================================

def _create_default_dict_list():
    """Provides a pickleable factory for creating a defaultdict of lists."""
    return defaultdict(list)

class MetaLearner:
    """Learns and selects optimal strategies for orchestration tasks.

    This class maintains a history of task performance and uses an epsilon-greedy
    strategy to balance exploiting the best-known strategies with exploring
    new ones. This allows the system to adapt its approach over time based on
    what has worked for similar projects in the past.
    """

    def __init__(self, epsilon: float = 0.3):
        """Initializes the MetaLearner.

        Args:
            epsilon (float): The exploration rate for the epsilon-greedy
                strategy. A value of 0.3 means a 30% chance of exploring a
                random strategy instead of exploiting the best-known one.
        """
        self.epsilon = epsilon
        self.task_history: List[TaskMetadata] = []
        self.strategy_performance: Dict[str, Dict[str, List[float]]] = defaultdict(_create_default_dict_list)
        self.available_strategies = ["high_quality", "high_speed", "balanced"]
        self.rl_q_table: Dict[str, Dict[str, float]] = {}

    def record_task(self, task: TaskMetadata):
        """Records the outcome of a completed task to refine its strategy.

        Args:
            task (TaskMetadata): A TaskMetadata object containing the details
                and success metrics of the completed task. This data is used
                to update the performance history of different strategies.
        """
        self.task_history.append(task)
        strategy_key = self._infer_strategy(task.optimal_config)
        self.strategy_performance[task.project_type][strategy_key].append(
            task.success_metrics.get('overall_quality', 0.0)
        )

    def get_optimal_strategy(self, project_type: str, complexity: float) -> Dict[str, Any]:
        """Selects a strategy for a new task using an epsilon-greedy approach.

        With probability epsilon, it will explore a random strategy. Otherwise,
        it will exploit the strategy with the best historical performance for
        the given project type.

        Args:
            project_type (str): The type of the project (e.g., 'data_pipeline')
                for which to select a strategy.
            complexity (float): The complexity of the project, used to scale
                strategy parameters.

        Returns:
            Dict[str, Any]: A dictionary defining the selected strategy's
                parameters (e.g., {'ga_generations': 10, 'rl_episodes': 5}).
        """
        # 1. Decide whether to explore or exploit
        if random.random() < self.epsilon:
            print("  [MetaLearner] Exploring a random strategy.")
            strategy_name = random.choice(self.available_strategies)
            return self._decode_strategy(strategy_name, complexity)

        # 2. If exploiting, check for existing knowledge
        if project_type not in self.strategy_performance:
            print("  [MetaLearner] No history for project type, using default strategy.")
            return self._default_strategy(complexity)

        # 3. Exploit the best-known strategy
        print("  [MetaLearner] Exploiting the best-known strategy.")
        best_strategy = "balanced"
        best_performance = float('-inf')

        for strategy, performances in self.strategy_performance[project_type].items():
            avg_performance = sum(performances) / len(performances)
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_strategy = strategy

        return self._decode_strategy(best_strategy, complexity)

    def _infer_strategy(self, config: Dict[str, Any]) -> str:
        """Infers the name of a strategy from its configuration parameters.

        Args:
            config: A dictionary of strategy parameters.

        Returns:
            The string name of the strategy.
        """
        ga_gens = config.get('ga_generations', 10)
        rl_eps = config.get('rl_episodes', 5)
        if ga_gens > 20 and rl_eps > 10:
            return "high_quality"
        elif ga_gens < 8 and rl_eps < 4:
            return "high_speed"
        return "balanced"

    def _decode_strategy(self, strategy: str, complexity: float) -> Dict[str, Any]:
        """Gets a configuration dictionary for a given strategy name.

        Args:
            strategy: The name of the strategy.
            complexity: The complexity of the project.

        Returns:
            A dictionary of parameters for the given strategy.
        """
        strategies = {
            "high_quality": {'ga_generations': int(25 * complexity), 'rl_episodes': int(12 * complexity)},
            "high_speed": {'ga_generations': 5, 'rl_episodes': 3},
            "balanced": {'ga_generations': int(15 * complexity), 'rl_episodes': int(8 * complexity)}
        }
        return strategies.get(strategy, strategies["balanced"])

    def _default_strategy(self, complexity: float) -> Dict[str, Any]:
        """Gets a default strategy for unknown project types.

        Args:
            complexity: The complexity of the project.

        Returns:
            A dictionary of parameters for the default strategy.
        """
        return {'ga_generations': int(10 * complexity), 'rl_episodes': int(5 * complexity)}

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
        except SyntaxError:
            return metrics  # No point in trying to run if syntax is wrong

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(script_code)
            temp_filename = temp_file.name

        try:
            # Execute the script as a subprocess
            result = subprocess.run(
                ['python3', temp_filename],
                capture_output=True,
                text=True,
                timeout=SINGLE_SCRIPT_VERIFICATION_TIMEOUT
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
