---
# Crispo: A Template-Based Code Generation System

Crispo is a system that generates Python scripts from high-level objectives. It uses a simple, template-based approach to create code for a variety of tasks, including data processing and the implementation of Learning-Augmented Algorithms (LAA).

## 🎯 System Overview

Crispo is designed to be a straightforward and transparent code generation tool. It takes a user's objective, such as "fetch data from an API," and matches it to a pre-defined code template. This template is then used to generate a Python script that accomplishes the desired task.

For more complex tasks, such as the co-design of Learning-Augmented Algorithms, Crispo can generate a "Solution Package" consisting of two scripts:

1.  **A Predictor Script:** An ML model that learns from historical data to predict future values.
2.  **An Algorithm Script:** A Learning-Augmented Algorithm that uses the ML prediction to make informed decisions.

## 🏗️ Core Architecture

Crispo's architecture is simple and direct. It consists of two main components:

1.  **`CodeGenerator`:** This is the heart of the system. It contains a set of pre-defined code templates and a simple logic for selecting the most appropriate template based on keywords in the user's objective.
2.  **`Verifier`:** This component is responsible for ensuring the quality and security of the generated code. It checks for syntax and runtime errors and executes the code in a sandboxed environment with strict resource limits to prevent security vulnerabilities.

## ✨ Key Features

### 1. Template-Based Code Generation

Crispo's primary feature is its simple and transparent template-based approach to code generation. This makes it easy to understand how the system works and to extend it with new templates.

### 2. Learning-Augmented Algorithm (LAA) Co-Design

Crispo can generate a complete "Solution Package" for Learning-Augmented Algorithms, including a predictor and an algorithm that work together.

### 3. Security-Focused Verification

The `Verifier` ensures that all generated code is executed in a secure, sandboxed environment with strict CPU and memory limits. This mitigates the risk of security vulnerabilities such as arbitrary code execution and resource exhaustion.

### 4. Solution Registry

Verified solutions are automatically versioned and saved to the `solution_registry/` directory. This creates a persistent, queryable knowledge base of high-quality solutions.

## Usage

Crispo is a command-line tool. The main entry point is `crispo.py`.

### Basic Example

```bash
python3 crispo.py --project "MyDataPipeline" --objective "Fetch data from an API, process it with pandas, and analyze with numpy"
```

### LAA Co-Design Example

To generate a Learning-Augmented Algorithm for the ski rental problem:

```bash
python3 crispo.py --project "SkiRentalLAA" --objective "Generate a learning-augmented algorithm for the ski rental problem" --trust-parameter 0.7
```

*Note: This requires a `ski_rental_history.csv` file in the root directory.*

## Licensing

`crispo` is licensed under the AGPLv3. For use in a closed-source or commercial application, a separate commercial license is required. Please contact `crispo.contact@gmail.com` for more information.

## Testing

The project uses the built-in `unittest` framework. To run the full test suite:

```bash
python3 -m unittest test_crispo.py
```
