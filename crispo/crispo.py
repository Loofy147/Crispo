#!/usr/bin/env python3
"""
OrchestratorAI: Autonomous Multi-Layer Script Orchestration System
"""

import argparse
import pickle
import random
from typing import Dict, List, Any
from datetime import datetime

from .crispo_core import (
    OrchestrationContext,
    CodeGenerator,
    TaskMetadata,
    Verifier,
    SkiRentalContext,
    OneMaxSearchContext
)
from .solution_registry import query_registry

# ============================================================================
# CRISPO ENGINE
# ============================================================================

class Crispo:
    """Main orchestration engine for the Crispo system.

    This class integrates all the core components to execute the
    end-to-end script generation and optimization pipeline.
    """

    def __init__(self, context: OrchestrationContext, problem_context=None):
        """Initializes the Crispo engine.

        Args:
            context (OrchestrationContext): The global context for this
                orchestration run, containing the project name and objective.
            problem_context: An optional problem-specific context for LAA tasks.
        """
        self.context = context
        self.code_generator = CodeGenerator()
        self.verifier = Verifier()
        self.problem_context = problem_context

    def orchestrate(
        self,
        complexity: float,
        trust_parameter: float,
    ) -> List[str]:
        """Executes the full, multi-phase orchestration pipeline.

        This method guides the process from strategy selection through code
        generation, optimization, and verification.

        Args:
            complexity (float): The complexity of the task, from 0.0 to 1.0.
            trust_parameter (float): The trust parameter (lambda) for
                learning-augmented algorithms.

        Returns:
            List[str]: A list of strings, where each string is the generated
                Python code for a single layer of the pipeline.
        """
        print("🚀 " + "="*68)
        print("ORCHESTRATOR AI: AUTONOMOUS EXECUTION")
        print(f"Project: {self.context.project}")
        print(f"Objective: {self.context.objective}")
        print("="*70)

        # Phase 1: Generate and Execute Pipeline
        print("\n💻 PHASE 1: Pipeline Generation and Execution")
        generated_scripts = []

        objective = self.context.objective.lower()
        if "ski rental" in objective or "one-max search" in objective:
            print("  - Generating a complete LAA Solution Package (Algorithm + Predictor)...")
            algorithm_script = self.code_generator.generate(0, self.context.objective, complexity, trust_parameter)
            predictor_script = self.code_generator._generate_predictor_template()

            with open("generated_algorithm.py", "w") as f:
                f.write(algorithm_script)
            with open("generated_predictor.py", "w") as f:
                f.write(predictor_script)

            generated_scripts = [algorithm_script, predictor_script]
            print("  - Solution Package saved to generated_algorithm.py and generated_predictor.py")
        else:
            for i in range(3): # Generate 3 layers for a standard pipeline
                script = self.code_generator.generate(i, self.context.objective, complexity, trust_parameter)
                generated_scripts.append(script)
                print(f"  - Generated Layer {i}")

        # Phase 2: Verification and Feedback
        print("\n🔬 PHASE 2: Verification and Feedback")

        if self.problem_context:
            laa_metrics = self.verifier.evaluate_learning_augmented_algorithm(
                algorithm_script_path="generated_algorithm.py",
                predictor_script_path="generated_predictor.py",
                trust_parameter=trust_parameter,
                problem_context=self.problem_context
            )
            print(f"  - Co-Designed Solution Competitive Ratio: {laa_metrics.get('competitive_ratio', 0.0):.2f}")
        else:
            metrics = self.verifier.verify_pipeline(generated_scripts)
            final_quality = metrics['overall_quality']
            print(f"  - Final Aggregated Quality: {final_quality:.2f}")

        return generated_scripts

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Command-line entry point for the Crispo system."""
    parser = argparse.ArgumentParser(description="Crispo: Autonomous Co-Design of ML Predictors and Algorithms")
    parser.add_argument("--project", type=str, default="AutoCode_Genesis", help="Project name.")
    parser.add_argument("--objective", type=str, default="Generate a self-optimizing multi-layer data processing script", help="The main objective.")
    parser.add_argument("--complexity", type=float, default=0.8, help="Complexity of the project (0.0 to 1.0).")
    parser.add_argument("--trust-parameter", type=float, default=0.8, help="Trust parameter (lambda) for the learning-augmented algorithm.")
    parser.add_argument("--query-registry", type=str, help="Query the solution registry. E.g., 'competitive_ratio:1.5'")

    args = parser.parse_args()

    if args.query_registry:
        try:
            metric, threshold_str = args.query_registry.split(':')
            threshold = float(threshold_str)
            results = query_registry(metric, threshold)

            print("\n" + "="*70)
            print("SOLUTION REGISTRY QUERY RESULTS")
            print(f"Solutions with {metric} <= {threshold}:")
            print("="*70)
            if not results:
                print("No solutions found matching the criterion.")
            else:
                for res in results:
                    print(f"  - Problem: {res['problem_type']}, Version: {res['version']}, Metrics: {res['metrics']}")
            print("="*70)
            return
        except ValueError:
            print("Invalid query format. Please use 'metric:value', e.g., 'competitive_ratio:1.5'")
            return

    context = OrchestrationContext(
        project=args.project,
        objective=args.objective
    )

    crispo = Crispo(context)

    final_scripts = crispo.orchestrate(
        complexity=args.complexity,
        trust_parameter=args.trust_parameter
    )

    print("\n" + "="*70)
    print("ORCHESTRATION COMPLETE")
    print("="*70)

    for i, script in enumerate(final_scripts):
        print(f"\n--- Layer {i} ---\n")
        print(script)

if __name__ == "__main__":
    main()
