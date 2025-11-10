#!/usr/bin/env python3
"""
Crispo: A Template-Based Code Generation System
"""

import argparse
from typing import List, Any

from .crispo_core import (
    OrchestrationContext,
    CodeGenerator,
    Verifier,
    SkiRentalContext,
    OneMaxSearchContext
)
from .solution_registry import query_registry

# ============================================================================
# CRISPO ENGINE
# ============================================================================

class Crispo:
    """Main orchestration engine for the Crispo system."""

    def __init__(self, context: OrchestrationContext, problem_context: Any = None):
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

    def orchestrate(self, trust_parameter: float) -> List[str]:
        """Executes the code generation and verification pipeline.

        Args:
            trust_parameter (float): The trust parameter (lambda) for
                learning-augmented algorithms.

        Returns:
            List[str]: A list of strings, where each string is the generated
                Python code for a single layer of the pipeline.
        """
        print("🚀 " + "="*68)
        print("CRISPO: TEMPLATE-BASED CODE GENERATION")
        print(f"Project: {self.context.project}")
        print(f"Objective: {self.context.objective}")
        print("="*70)

        # Phase 1: Generate and Execute Pipeline
        print("\n💻 PHASE 1: Pipeline Generation and Execution")
        generated_scripts = []

        objective = self.context.objective.lower()
        if "ski rental" in objective or "one-max search" in objective:
            print("  - Generating a complete LAA Solution Package (Algorithm + Predictor)...")
            algorithm_script = self.code_generator.generate(self.context.objective, trust_parameter)
            predictor_script = self.code_generator._generate_predictor_template()

            with open("generated_algorithm.py", "w") as f:
                f.write(algorithm_script)
            with open("generated_predictor.py", "w") as f:
                f.write(predictor_script)

            generated_scripts = [algorithm_script, predictor_script]
            print("  - Solution Package saved to generated_algorithm.py and generated_predictor.py")
        else:
            # For standard pipelines, generate a sequence of scripts
            # This is a simplified example; a real implementation might have more sophisticated logic
            script1 = self.code_generator.generate(self.context.objective)
            generated_scripts.append(script1)
            print("  - Generated Script 1")

        # Phase 2: Verification
        print("\n🔬 PHASE 2: Verification")

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
    parser = argparse.ArgumentParser(description="Crispo: A Template-Based Code Generation System")
    parser.add_argument("--project", type=str, default="AutoCode_Genesis", help="Project name.")
    parser.add_argument("--objective", type=str, default="Generate a data processing script", help="The main objective.")
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
        except ValueError:
            print("Invalid query format. Please use 'metric:value', e.g., 'competitive_ratio:1.5'")
        return

    context = OrchestrationContext(
        project=args.project,
        objective=args.objective
    )

    problem_context = None
    objective = args.objective.lower()
    if "ski rental" in objective:
        problem_context = SkiRentalContext()
    elif "one-max search" in objective:
        problem_context = OneMaxSearchContext()

    crispo = Crispo(context, problem_context=problem_context)

    final_scripts = crispo.orchestrate(
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
