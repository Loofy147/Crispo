"""
Predictor Quality Evaluation Component
This module provides the PredictorEvaluator class, which is responsible for
assessing the quality of the generated predictor scripts, particularly their
Uncertainty Quantification (UQ) calibration.
"""
import numpy as np
from typing import List, Tuple
import pandas as pd

class PredictorEvaluator:
    """Evaluates the quality of a generated predictor."""

    def evaluate_uq_calibration(self, predictor, test_data: List[Tuple[int, float]]):
        """
        Evaluates the UQ calibration of a time-series predictor.
        It assesses whether the predictor's uncertainty intervals are reliable.
        For a 95% confidence interval, we expect 95% of true values to fall
        within the predicted range.

        Args:
            predictor: An instantiated predictor object with a `predict_interval` method.
            test_data (List[Tuple[int, float]]): A list of (index, value) pairs
                representing the hold-out test set.

        Returns:
            A dictionary containing quality metrics:
            - 'coverage_rate': The proportion of true values that fell within the
                               predicted uncertainty interval.
            - 'interval_sharpness': The average width of the prediction intervals.
                                    Narrower is generally better, assuming good
                                    coverage.
        """
        coverage = []
        intervals = []

        # For a time-series model, we can't just pass `x`. We need to simulate
        # predicting the next step in a sequence.
        # The `test_data` represents future values the model hasn't seen.
        # The predictor passed in has been trained on data *before* this test set.

        for i, (idx, y_true) in enumerate(test_data):
            # Predict one step ahead from the last known point.
            # In a real scenario, we might re-train, but for this evaluation,
            # we'll just predict farther into the future.
            predicted_interval = predictor.predict_interval(steps=i + 1)
            lower, upper = predicted_interval[0], predicted_interval[1]

            coverage.append(lower <= y_true <= upper)
            intervals.append((lower, upper))

        if not coverage:
            return {'coverage_rate': 0.0, 'interval_sharpness': float('inf')}

        coverage_rate = np.mean(coverage)
        avg_sharpness = np.mean([upper - lower for lower, upper in intervals])

        return {
            'coverage_rate': coverage_rate,
            'interval_sharpness': avg_sharpness
        }
