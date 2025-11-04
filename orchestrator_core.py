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

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class TaskMetadata:
    """Metadata about orchestration tasks for meta-learning."""
    task_id: str
    project_type: str
    complexity_level: float
    domain: str
    success_metrics: Dict[str, float]
    optimal_config: Dict[str, Any]
    timestamp: str

@dataclass
class OrchestrationContext:
    """Global context for a single orchestration run."""
    project: str
    objective: str
    feedback_loop: Dict = field(default_factory=dict)
    resource_usage: Dict = field(default_factory=dict)
    failure_cases: List[str] = field(default_factory=list)

@dataclass
class ScriptToken:
    """Represents a token in the script generation vocabulary."""
    value: str
    weight: float = 1.0

@dataclass
class LayerParameters:
    """Parameters for a script generation layer."""
    layer_id: int
    weights: Dict[str, float]
    biases: Dict[str, float]
    temperature: float = 1.0

    def clone(self):
        """Create a deep copy for genetic algorithms."""
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
    """Evolves layer configurations using a genetic algorithm."""

    def __init__(self, population_size: int = 20, mutation_rate: float = 0.15, crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def execute(self, template_params: LayerParameters, context: Dict, generations: int = 10) -> LayerParameters:
        """Run the genetic algorithm."""
        population = self._initialize_population(template_params)

        for gen in range(generations):
            fitness_scores = [self._evaluate_fitness(ind, context) for ind in population]
            
            print(f"  [GA] Gen {gen + 1}: Best Fitness={max(fitness_scores):.3f}, Avg Fitness={sum(fitness_scores)/len(fitness_scores):.3f}")

            parents = self._tournament_selection(population, fitness_scores)
            
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self._crossover(parents[i], parents[i+1])
                offspring.extend([child1, child2])

            offspring = [self._mutate(child) for child in offspring]

            # Elitism: keep the best individual
            best_individual = population[fitness_scores.index(max(fitness_scores))]

            # Create a new population with the best individual and the offspring
            new_population = [best_individual]
            new_population.extend(offspring)

            # Fill the rest of the population with individuals from the old population
            # sorted by fitness
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

            fill_count = self.population_size - len(new_population)
            if fill_count > 0:
                new_population.extend(sorted_population[:fill_count])

            population = new_population[:self.population_size]


        final_fitness = [self._evaluate_fitness(ind, context) for ind in population]
        best_idx = final_fitness.index(max(final_fitness))
        return population[best_idx]

    def _initialize_population(self, template: LayerParameters) -> List[LayerParameters]:
        """Create an initial random population."""
        population = []
        for _ in range(self.population_size):
            individual = template.clone()
            for key in individual.weights:
                individual.weights[key] *= random.uniform(0.5, 1.5)
            for key in individual.biases:
                individual.biases[key] += random.uniform(-0.2, 0.2)
            individual.temperature = random.uniform(0.7, 1.3)
            population.append(individual)
        return population

    def _evaluate_fitness(self, params: LayerParameters, context: Dict) -> float:
        """Fitness function. Higher is better."""
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
        """Select parents via tournament."""
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
        """Single-point crossover."""
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
        """Randomly mutate parameters."""
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
    """Q-learning agent for parameter fine-tuning."""

    def __init__(self, learning_rate: float = 0.1, discount: float = 0.95, epsilon: float = 0.2):
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_table: Dict[str, Dict[str, float]] = {}

    def execute(self, initial_params: LayerParameters, context: Dict, episodes: int = 5) -> LayerParameters:
        """Execute RL fine-tuning."""
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

            print(f"  [RL] Episode {episode + 1}: Reward={episode_reward:.3f}, Temp={params.temperature:.2f}")

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_params = params.clone()
        
        return best_params

    def _encode_state(self, params: LayerParameters, context: Dict) -> str:
        """Encode state as a string for Q-table lookup."""
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
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon or state not in self.q_table or not self.q_table[state]:
            return {
                'weight_delta': random.uniform(-0.1, 0.1),
                'bias_delta': random.uniform(-0.05, 0.05),
                'temp_delta': random.uniform(-0.1, 0.1)
            }
        best_action_str = max(self.q_table[state], key=self.q_table[state].get)
        return json.loads(best_action_str)

    def _apply_action(self, params: LayerParameters, action: Dict) -> LayerParameters:
        """Apply action to parameters."""
        new_params = params.clone()
        for key in new_params.weights:
            new_params.weights[key] += action['weight_delta']
        for key in new_params.biases:
            new_params.biases[key] += action['bias_delta']
        new_params.temperature += action['temp_delta']
        return new_params

    def _calculate_reward(self, params: LayerParameters) -> float:
        """Calculate reward for the current parameters."""
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
        """Update the Q-table."""
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
    """Multi-head attention for inter-layer communication."""

    def __init__(self, embed_dim: int = 16, num_heads: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def execute(self, query: List[float], keys: List[List[float]], values: List[List[float]]) -> Dict:
        """Execute the attention mechanism."""
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
    """Generates multi-layer scripts with intent-based template selection."""

    def generate(self, params: LayerParameters, layer_id: int, objective: str) -> str:
        """
        Selects and generates a script for a single layer based on the
        pipeline's objective and the current layer ID.
        """
        objective = objective.lower()

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

    def _generate_high_complexity_template(self, params: LayerParameters, layer_id: int) -> str:
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
    # Example usage with a mock API endpoint
    mock_input_context = {{'api_endpoint': 'https://jsonplaceholder.typicode.com/todos/1'}}
    result = process_layer_{layer_id}(mock_input_context)
    print(json.dumps(result))
'''

    def _generate_simple_template(self, params: LayerParameters, layer_id: int) -> str:
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
    """Helper function for pickling defaultdict."""
    return defaultdict(list)

class MetaLearner:
    """Learns which optimization strategies work best for different task types."""

    def __init__(self, epsilon: float = 0.3):
        self.epsilon = epsilon
        self.task_history: List[TaskMetadata] = []
        self.strategy_performance: Dict[str, Dict[str, List[float]]] = defaultdict(_create_default_dict_list)
        self.available_strategies = ["high_quality", "high_speed", "balanced"]

    def record_task(self, task: TaskMetadata):
        """Record a task's execution for meta-learning."""
        self.task_history.append(task)
        strategy_key = self._infer_strategy(task.optimal_config)
        self.strategy_performance[task.project_type][strategy_key].append(
            task.success_metrics.get('overall_quality', 0.0)
        )

    def get_optimal_strategy(self, project_type: str, complexity: float) -> Dict[str, Any]:
        """
        Get the optimal strategy for a new task using an epsilon-greedy approach.
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
        """Infer the strategy from a configuration."""
        ga_gens = config.get('ga_generations', 10)
        rl_eps = config.get('rl_episodes', 5)
        if ga_gens > 20 and rl_eps > 10:
            return "high_quality"
        elif ga_gens < 8 and rl_eps < 4:
            return "high_speed"
        return "balanced"

    def _decode_strategy(self, strategy: str, complexity: float) -> Dict[str, Any]:
        """Get a configuration from a strategy."""
        strategies = {
            "high_quality": {'ga_generations': int(25 * complexity), 'rl_episodes': int(12 * complexity)},
            "high_speed": {'ga_generations': 5, 'rl_episodes': 3},
            "balanced": {'ga_generations': int(15 * complexity), 'rl_episodes': int(8 * complexity)}
        }
        return strategies.get(strategy, strategies["balanced"])

    def _default_strategy(self, complexity: float) -> Dict[str, Any]:
        """Get a default strategy for unknown project types."""
        return {'ga_generations': int(10 * complexity), 'rl_episodes': int(5 * complexity)}

# ============================================================================
# VERIFICATION UNIT
# ============================================================================

class Verifier:
    """Verifies the quality of a generated script."""

    def verify_script(self, script_code: str) -> Dict[str, float]:
        """
        Verifies a script by saving it to a temporary file and executing it
        as a subprocess. This provides a high-fidelity check for both syntax
        and runtime errors.
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
                timeout=10 # Add a timeout to prevent long-running scripts
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
        """
        Verifies a full pipeline by executing scripts sequentially and passing
        the context between them.
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
                    timeout=15
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
