import json
import random
import math
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import re

# ============================================================================
# PART 1: CORE STRUCTURES (Foundation)
# ============================================================================

@dataclass
class ScriptToken:
    """Represents a token in the script generation vocabulary"""
    value: str
    weight: float = 1.0
    embedding: List[float] = field(default_factory=lambda: [random.gauss(0, 0.1) for _ in range(16)])
    frequency: int = 0
    
    def activate(self, bias: float, activation_type: str = "relu") -> float:
        """Apply activation function with bias"""
        x = self.weight + bias
        
        if activation_type == "relu":
            return max(0, x)
        elif activation_type == "sigmoid":
            return 1 / (1 + math.exp(-x))
        elif activation_type == "tanh":
            return math.tanh(x)
        else:
            return x

@dataclass
class LayerParameters:
    """Parameters for a script generation layer"""
    layer_id: int
    weights: Dict[str, float]
    biases: Dict[str, float]
    tokens: Dict[str, ScriptToken]
    temperature: float = 1.0
    dropout: float = 0.0
    
    # Attention parameters (Part 3)
    attention_heads: int = 4
    query_weights: List[List[float]] = field(default_factory=list)
    key_weights: List[List[float]] = field(default_factory=list)
    value_weights: List[List[float]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.query_weights:
            dim = 16
            self.query_weights = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(self.attention_heads)]
            self.key_weights = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(self.attention_heads)]
            self.value_weights = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(self.attention_heads)]
    
    def apply_weights(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply layer weights to input data"""
        output = {}
        for key, value in input_data.items():
            weight = self.weights.get(key, 1.0)
            bias = self.biases.get(key, 0.0)
            
            if isinstance(value, (int, float)):
                output[key] = (value * weight + bias) * self.temperature
            else:
                output[key] = value
        
        return output
    
    def tokenize(self, code_template: str) -> List[ScriptToken]:
        """Convert code template into weighted tokens"""
        tokens = []
        raw_tokens = re.findall(r'\w+|[^\w\s]', code_template)
        
        for token_value in raw_tokens:
            if token_value in self.tokens:
                token = self.tokens[token_value]
                token.frequency += 1
            else:
                token = ScriptToken(value=token_value)
                self.tokens[token_value] = token
            
            tokens.append(token)
        
        return tokens
    
    def clone(self):
        """Create a deep copy for genetic algorithms"""
        return LayerParameters(
            layer_id=self.layer_id,
            weights=self.weights.copy(),
            biases=self.biases.copy(),
            tokens={k: ScriptToken(v.value, v.weight, v.embedding.copy(), v.frequency) 
                   for k, v in self.tokens.items()},
            temperature=self.temperature,
            dropout=self.dropout,
            attention_heads=self.attention_heads,
            query_weights=[q.copy() for q in self.query_weights],
            key_weights=[k.copy() for k in self.key_weights],
            value_weights=[v.copy() for v in self.value_weights]
        )

# ============================================================================
# PART 2: REINFORCEMENT LEARNING (Parameter Optimization)
# ============================================================================

@dataclass
class RLState:
    """State representation for RL agent"""
    layer_id: int
    context: Dict[str, Any]
    previous_rewards: List[float]
    code_metrics: Dict[str, float]
    
    def to_vector(self) -> List[float]:
        """Convert state to feature vector"""
        features = [
            self.layer_id,
            self.context.get('complexity', 0.5),
            self.context.get('temperature', 1.0),
            len(self.previous_rewards),
            sum(self.previous_rewards[-5:]) / max(1, len(self.previous_rewards[-5:])),  # Avg recent reward
            self.code_metrics.get('length', 0) / 1000,  # Normalized length
            self.code_metrics.get('quality', 0.5),
        ]
        return features

@dataclass
class RLAction:
    """Action space for parameter adjustment"""
    weight_delta: Dict[str, float]
    bias_delta: Dict[str, float]
    temperature_delta: float
    
    @staticmethod
    def sample_random() -> 'RLAction':
        """Sample random action"""
        return RLAction(
            weight_delta={
                'complexity': random.uniform(-0.1, 0.1),
                'execution': random.uniform(-0.1, 0.1),
                'optimization': random.uniform(-0.1, 0.1),
            },
            bias_delta={
                'logging': random.uniform(-0.05, 0.05),
                'safety': random.uniform(-0.05, 0.05),
            },
            temperature_delta=random.uniform(-0.1, 0.1)
        )

class QLearningAgent:
    """Q-Learning agent for parameter optimization"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, epsilon: float = 0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: state_hash -> {action_hash: q_value}
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.state_history: List[Tuple[str, RLAction, float]] = []
    
    def get_state_hash(self, state: RLState) -> str:
        """Hash state for Q-table lookup"""
        state_vec = state.to_vector()
        # Discretize continuous values
        discretized = tuple(round(v, 1) for v in state_vec)
        return str(discretized)
    
    def get_action_hash(self, action: RLAction) -> str:
        """Hash action for Q-table lookup"""
        return str((
            tuple(sorted(action.weight_delta.items())),
            tuple(sorted(action.bias_delta.items())),
            round(action.temperature_delta, 2)
        ))
    
    def select_action(self, state: RLState) -> RLAction:
        """Epsilon-greedy action selection"""
        state_hash = self.get_state_hash(state)
        
        # Exploration
        if random.random() < self.epsilon:
            return RLAction.sample_random()
        
        # Exploitation - choose best known action
        if state_hash in self.q_table and self.q_table[state_hash]:
            best_action_hash = max(self.q_table[state_hash], key=self.q_table[state_hash].get)
            # Reconstruct action (simplified - in practice, store actual actions)
            return RLAction.sample_random()  # Placeholder
        
        return RLAction.sample_random()
    
    def update_q_value(self, state: RLState, action: RLAction, reward: float, next_state: RLState):
        """Q-learning update rule"""
        state_hash = self.get_state_hash(state)
        action_hash = self.get_action_hash(action)
        next_state_hash = self.get_state_hash(next_state)
        
        # Initialize Q-table entries
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}
        if action_hash not in self.q_table[state_hash]:
            self.q_table[state_hash][action_hash] = 0.0
        
        # Get max Q-value for next state
        max_next_q = 0.0
        if next_state_hash in self.q_table and self.q_table[next_state_hash]:
            max_next_q = max(self.q_table[next_state_hash].values())
        
        # Q-learning update
        current_q = self.q_table[state_hash][action_hash]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_hash][action_hash] = new_q
        
        self.state_history.append((state_hash, action, reward))
    
    def apply_action(self, params: LayerParameters, action: RLAction):
        """Apply action to modify parameters"""
        for key, delta in action.weight_delta.items():
            if key in params.weights:
                params.weights[key] = max(0.1, min(2.0, params.weights[key] + delta))
        
        for key, delta in action.bias_delta.items():
            if key in params.biases:
                params.biases[key] = max(-0.5, min(0.5, params.biases[key] + delta))
        
        params.temperature = max(0.5, min(2.0, params.temperature + action.temperature_delta))

# ============================================================================
# PART 3: ATTENTION MECHANISMS (Layer Communication)
# ============================================================================

class MultiHeadAttention:
    """Transformer-style attention between layers"""
    
    def __init__(self, embed_dim: int = 16, num_heads: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
    
    def compute_attention(self, query: List[float], keys: List[List[float]], 
                         values: List[List[float]]) -> List[float]:
        """
        Compute scaled dot-product attention
        query: Current layer embedding [embed_dim]
        keys: Previous layers embeddings [num_layers, embed_dim]
        values: Previous layers embeddings [num_layers, embed_dim]
        """
        if not keys or not values:
            return query
        
        # Compute attention scores
        scores = []
        for key in keys:
            score = sum(q * k for q, k in zip(query, key)) / math.sqrt(self.embed_dim)
            scores.append(score)
        
        # Softmax normalization
        exp_scores = [math.exp(s) for s in scores]
        sum_exp = sum(exp_scores)
        attention_weights = [s / sum_exp for s in exp_scores]
        
        # Weighted sum of values
        output = [0.0] * self.embed_dim
        for weight, value in zip(attention_weights, values):
            for i in range(self.embed_dim):
                output[i] += weight * value[i]
        
        return output, attention_weights
    
    def multi_head_attention(self, query_weights: List[List[float]], 
                            key_weights: List[List[float]],
                            value_weights: List[List[float]],
                            query: List[float], 
                            keys: List[List[float]], 
                            values: List[List[float]]) -> List[float]:
        """Multi-head attention computation"""
        head_outputs = []
        
        for head in range(self.num_heads):
            # Project query, keys, values for this head
            q_proj = self._project(query, query_weights[head])
            k_proj = [self._project(k, key_weights[head]) for k in keys]
            v_proj = [self._project(v, value_weights[head]) for v in values]
            
            # Compute attention for this head
            head_out, _ = self.compute_attention(q_proj, k_proj, v_proj)
            head_outputs.append(head_out)
        
        # Concatenate heads
        concatenated = []
        for head_out in head_outputs:
            concatenated.extend(head_out[:self.head_dim])
        
        return concatenated[:self.embed_dim]
    
    def _project(self, vector: List[float], weights: List[float]) -> List[float]:
        """Linear projection"""
        return [sum(v * w for v, w in zip(vector, weights[:len(vector)])) for _ in range(self.head_dim)]

# ============================================================================
# PART 4: GENETIC ALGORITHMS (Configuration Evolution)
# ============================================================================

class GeneticOptimizer:
    """Genetic algorithm for evolving layer configurations"""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.15, 
                 crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
    
    def initialize_population(self, template_params: LayerParameters) -> List[LayerParameters]:
        """Create initial random population"""
        population = []
        for _ in range(self.population_size):
            individual = template_params.clone()
            
            # Randomize weights
            for key in individual.weights:
                individual.weights[key] *= random.uniform(0.5, 1.5)
            
            # Randomize biases
            for key in individual.biases:
                individual.biases[key] += random.uniform(-0.2, 0.2)
            
            # Randomize temperature
            individual.temperature = random.uniform(0.7, 1.3)
            
            population.append(individual)
        
        return population
    
    def evaluate_fitness(self, params: LayerParameters, test_context: Dict[str, Any]) -> float:
        """
        Fitness function for layer parameters
        Higher is better
        """
        fitness = 0.0
        
        # Reward balanced weights (not too extreme)
        avg_weight = sum(params.weights.values()) / len(params.weights)
        weight_variance = sum((w - avg_weight) ** 2 for w in params.weights.values()) / len(params.weights)
        fitness += max(0, 1.0 - weight_variance)  # Prefer low variance
        
        # Reward moderate temperature
        ideal_temp = 1.0
        temp_penalty = abs(params.temperature - ideal_temp)
        fitness += max(0, 1.0 - temp_penalty)
        
        # Reward based on context alignment
        if 'desired_complexity' in test_context:
            complexity_alignment = 1.0 - abs(params.weights.get('complexity', 1.0) - test_context['desired_complexity'])
            fitness += complexity_alignment
        
        return max(0.0, fitness)
    
    def select_parents(self, population: List[LayerParameters], 
                      fitness_scores: List[float]) -> List[LayerParameters]:
        """Tournament selection"""
        parents = []
        tournament_size = 3
        
        for _ in range(len(population) // 2):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Winner is the one with highest fitness
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            parents.append(population[winner_idx])
        
        return parents
    
    def crossover(self, parent1: LayerParameters, parent2: LayerParameters) -> Tuple[LayerParameters, LayerParameters]:
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # Crossover weights
        weight_keys = list(parent1.weights.keys())
        crossover_point = random.randint(0, len(weight_keys))
        
        for i, key in enumerate(weight_keys):
            if i < crossover_point:
                child1.weights[key] = parent2.weights[key]
                child2.weights[key] = parent1.weights[key]
        
        # Crossover temperature
        if random.random() < 0.5:
            child1.temperature, child2.temperature = child2.temperature, child1.temperature
        
        return child1, child2
    
    def mutate(self, params: LayerParameters) -> LayerParameters:
        """Random mutation"""
        mutated = params.clone()
        
        # Mutate weights
        for key in mutated.weights:
            if random.random() < self.mutation_rate:
                mutated.weights[key] *= random.uniform(0.8, 1.2)
                mutated.weights[key] = max(0.1, min(2.0, mutated.weights[key]))
        
        # Mutate biases
        for key in mutated.biases:
            if random.random() < self.mutation_rate:
                mutated.biases[key] += random.uniform(-0.1, 0.1)
                mutated.biases[key] = max(-0.5, min(0.5, mutated.biases[key]))
        
        # Mutate temperature
        if random.random() < self.mutation_rate:
            mutated.temperature += random.uniform(-0.2, 0.2)
            mutated.temperature = max(0.5, min(2.0, mutated.temperature))
        
        return mutated
    
    def evolve(self, population: List[LayerParameters], test_context: Dict[str, Any], 
               generations: int = 10) -> LayerParameters:
        """Run genetic algorithm"""
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(ind, test_context) for ind in population]
            
            print(f"Generation {gen + 1}: Best Fitness = {max(fitness_scores):.3f}, "
                  f"Avg Fitness = {sum(fitness_scores) / len(fitness_scores):.3f}")
            
            # Selection
            parents = self.select_parents(population, fitness_scores)
            
            # Crossover
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self.crossover(parents[i], parents[i + 1])
                offspring.extend([child1, child2])
            
            # Mutation
            offspring = [self.mutate(child) for child in offspring]
            
            # Elitism - keep best individual
            best_idx = fitness_scores.index(max(fitness_scores))
            offspring[0] = population[best_idx]
            
            population = offspring
        
        # Return best individual
        final_fitness = [self.evaluate_fitness(ind, test_context) for ind in population]
        best_idx = final_fitness.index(max(final_fitness))
        return population[best_idx]

# ============================================================================
# PART 5: INTEGRATED NEURAL SCRIPT LAYER
# ============================================================================

class NeuralScriptLayer:
    """Enhanced layer with RL, Attention, and GA support"""
    
    def __init__(self, layer_id: int, parameters: LayerParameters, use_attention: bool = True):
        self.layer_id = layer_id
        self.parameters = parameters
        self.use_attention = use_attention
        
        self.attention = MultiHeadAttention(num_heads=parameters.attention_heads)
        self.embedding = [random.gauss(0, 0.1) for _ in range(16)]
        
        self.generated_code = ""
        self.generation_metrics = {}
    
    def forward_pass(self, input_context: Dict[str, Any], 
                    previous_layers: List['NeuralScriptLayer'] = None) -> str:
        """Generate code with attention to previous layers"""
        
        # Apply attention if previous layers exist
        attended_context = input_context.copy()
        
        if self.use_attention and previous_layers:
            keys = [layer.embedding for layer in previous_layers]
            values = [layer.embedding for layer in previous_layers]
            
            attended_embedding, attention_weights = self.attention.compute_attention(
                self.embedding, keys, values
            )
            
            # Use attention weights to blend context from previous layers
            attended_context['attention_influence'] = sum(attention_weights) / len(attention_weights)
            attended_context['attended_features'] = attended_embedding[:3]  # Sample features
        
        # Apply layer parameters
        weighted_context = self.parameters.apply_weights(attended_context)
        
        # Generate code
        code = self._generate_code(weighted_context)
        self.generated_code = code
        
        # Track metrics
        self.generation_metrics = {
            'length': len(code),
            'complexity': weighted_context.get('complexity', 0.5),
            'temperature': self.parameters.temperature
        }
        
        return code
    
    def _generate_code(self, context: Dict[str, Any]) -> str:
        """Generate code template"""
        layer_id = self.layer_id
        complexity = context.get('complexity', 1.0)
        purpose = context.get('purpose', 'processing')
        
        if complexity > 0.7:
            return f'''
# Layer {layer_id}: High Complexity Generator
class Layer{layer_id}System:
    def __init__(self, config):
        self.config = config
        self.weights = {dict(list(self.parameters.weights.items())[:2])}
        self.biases = {dict(list(self.parameters.biases.items())[:2])}
        self.temp = {self.parameters.temperature:.2f}
    
    def process(self, data):
        # Apply weighted transformation
        result = data
        for key, weight in self.weights.items():
            result = result * weight + self.biases.get(key, 0)
        return result * self.temp
    
    def generate_next_layer(self):
        code = """
# Layer {layer_id + 1}: Generated Output
def execute():
    return 'Layer {layer_id + 1} executed'
"""
        return code
'''
        else:
            return f'''
# Layer {layer_id}: Simple {purpose}
def process_{purpose}(input_val):
    weight = {self.parameters.weights.get('execution', 1.0):.2f}
    bias = {self.parameters.biases.get('execution', 0.0):.2f}
    return input_val * weight + bias
'''

# ============================================================================
# PART 6: UNIFIED SYSTEM
# ============================================================================

class AdaptiveMultiLayerSystem:
    """Complete system integrating RL, GA, and Attention"""
    
    def __init__(self, num_layers: int = 3):
        self.num_layers = num_layers
        self.layers: List[NeuralScriptLayer] = []
        
        # RL Agent for parameter optimization
        self.rl_agent = QLearningAgent()
        
        # GA Optimizer for configuration evolution
        self.ga_optimizer = GeneticOptimizer()
        
        self._initialize_layers()
    
    def _initialize_layers(self):
        """Initialize layers with default parameters"""
        for i in range(self.num_layers):
            params = LayerParameters(
                layer_id=i,
                weights={'complexity': 1.0, 'execution': 1.0, 'optimization': 1.0},
                biases={'logging': 0.0, 'safety': 0.0},
                tokens={},
                temperature=1.0,
                attention_heads=4
            )
            layer = NeuralScriptLayer(i, params, use_attention=True)
            self.layers.append(layer)
    
    def generate_with_rl(self, context: Dict[str, Any], episodes: int = 5) -> List[str]:
        """Generate code while learning optimal parameters via RL"""
        
        print("\n" + "="*70)
        print("REINFORCEMENT LEARNING MODE")
        print("="*70)
        
        best_reward = float('-inf')
        best_outputs = []
        
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            outputs = []
            total_reward = 0.0
            
            for i, layer in enumerate(self.layers):
                # Create RL state
                state = RLState(
                    layer_id=i,
                    context=context,
                    previous_rewards=[],
                    code_metrics=layer.generation_metrics
                )
                
                # Select action
                action = self.rl_agent.select_action(state)
                
                # Apply action to parameters
                self.rl_agent.apply_action(layer.parameters, action)
                
                # Generate code
                prev_layers = self.layers[:i] if i > 0 else None
                code = layer.forward_pass(context, prev_layers)
                outputs.append(code)
                
                # Calculate reward
                reward = self._calculate_reward(code, context)
                total_reward += reward
                
                # Update Q-values
                next_state = RLState(
                    layer_id=i,
                    context=context,
                    previous_rewards=[reward],
                    code_metrics=layer.generation_metrics
                )
                self.rl_agent.update_q_value(state, action, reward, next_state)
                
                print(f"  Layer {i}: Reward = {reward:.3f}, Temp = {layer.parameters.temperature:.2f}")
            
            print(f"  Total Reward: {total_reward:.3f}")
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_outputs = outputs
        
        return best_outputs
    
    def evolve_with_ga(self, context: Dict[str, Any], generations: int = 10):
        """Evolve layer configurations using genetic algorithms"""
        
        print("\n" + "="*70)
        print("GENETIC ALGORITHM MODE")
        print("="*70)
        
        for layer in self.layers:
            print(f"\nEvolving Layer {layer.layer_id}...")
            
            # Initialize population
            population = self.ga_optimizer.initialize_population(layer.parameters)
            
            # Evolve
            best_params = self.ga_optimizer.evolve(population, context, generations)
            
            # Update layer with evolved parameters
            layer.parameters = best_params
            
            print(f"  Optimized Weights: {layer.parameters.weights}")
            print(f"  Optimized Temperature: {layer.parameters.temperature:.2f}")
    
    def generate_with_attention(self, context: Dict[str, Any]) -> List[str]:
        """Generate code with full attention between layers"""
        
        print("\n" + "="*70)
        print("ATTENTION-BASED GENERATION")
        print("="*70)
        
        outputs = []
        
        for i, layer in enumerate(self.layers):
            prev_layers = self.layers[:i] if i > 0 else None
            
            print(f"\nLayer {i} (attending to {i} previous layers)")
            
            code = layer.forward_pass(context, prev_layers)
            outputs.append(code)
            
            print(f"  Generated {len(code)} characters")
            if i > 0:
                print(f"  Attention influence: {context.get('attention_influence', 0):.3f}")
        
        return outputs
    
    def _calculate_reward(self, code: str, context: Dict[str, Any]) -> float:
        """Calculate reward for RL"""
        reward = 0.0
        
        # Reward based on length (prefer moderate length)
        target_length = context.get('target_length', 500)
        length_diff = abs(len(code) - target_length)
        reward += max(0, 1.0 - length_diff / target_length)
        
        # Reward for code quality indicators
        if 'def ' in code or 'class ' in code:
            reward += 0.3
        if 'import ' in code:
            reward += 0.1
        
        return reward
    
    def full_pipeline(self, context: Dict[str, Any]):
        """Run complete pipeline: GA evolution → RL optimization → Attention generation"""
        
        print("\n" + "🚀 " + "="*68)
        print("FULL ADAPTIVE PIPELINE")
        print("="*70)
        
        # Step 1: Evolve configurations
        print("\n📊 STEP 1: Genetic Evolution")
        self.evolve_with_ga(context, generations=5)
        
        # Step 2: Fine-tune with RL
        print("\n🎯 STEP 2: Reinforcement Learning")
        rl_outputs = self.generate_with_rl(context, episodes=3)
        
        # Step 3: Generate with attention
        print("\n🧠 STEP 3: Attention-Based Generation")
        final_outputs = self.generate_with_attention(context)
        
        return final_outputs


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ADAPTIVE MULTI-LAYER SCRIPT SYSTEM")
    print("Integrating: RL + Genetic Algorithms + Attention Mechanisms")
    print("="*70)
    
    # Create system
    system = AdaptiveMultiLayerSystem(num_layers=3)
    
    # Define context
    context = {
        'language': 'python',
        'purpose': 'data_processing',
        'complexity': 0.8,
        'desired_complexity': 0.9,
        'target_length': 600,
        'temperature': 1.0
    }
    
    # Run full pipeline
    outputs = system.full_pipeline(context)
    
    # Display results
    print("\n" + "="*70)
    print("FINAL GENERATED CODE")
    print("="*70)
    
    for i, code in enumerate(outputs):
        print(f"\n{'='*70}")
        print(f"Layer {i} Output ({len(code)} chars):")
        print(f"{'='*70}")
        print(code[:400] + "..." if len(code) > 400 else code)