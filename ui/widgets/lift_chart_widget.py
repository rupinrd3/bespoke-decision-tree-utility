# utility/ui/widgets/lift_chart_widget.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lift Chart Widget for Bespoke Utility
Displays a lift chart for model evaluation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Any

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from utils.metrics_calculator import CentralMetricsCalculator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class LiftChartWidget(QWidget):
    """
    A widget to display a lift chart.
    The lift chart shows how much more likely we are to capture positive instances
    when using the model compared to random selection.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0,0,0,0)

        self._default_title = "Lift Chart"
        self._clear_plot()

    def _clear_plot(self):
        """Clears the plot and sets default labels."""
        self.ax.clear()
        self.ax.set_title(self._default_title)
        self.ax.set_xlabel("Percentage of Population (Deciles)")
        self.ax.set_ylabel("Lift")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.canvas.draw()

    def plot_lift_chart(self, y_true: Union[pd.Series, np.ndarray],
                        y_pred_proba: Union[pd.Series, np.ndarray],
                        positive_class_label: Any = 1,
                        n_bins: int = 10,
                        title: Optional[str] = None):
        """
        Generates and plots the lift chart according to standard lift chart logic.
        
        Implementation follows the standard approach:
        1. Sort observations by predicted probability (descending)
        2. Divide into equal-sized quantiles (typically deciles)
        3. Calculate response rate for each quantile
        4. Calculate lift as: (quantile response rate) / (overall response rate)
        
        A lift > 1 means the model performs better than random for that segment.
        A lift = 1 means the model performs as well as random.
        A lift < 1 means the model performs worse than random for that segment.

        Args:
            y_true: True binary labels (0 or 1, or matching positive_class_label).
            y_pred_proba: Predicted probabilities for the positive class.
            positive_class_label: The label of the positive class in y_true.
            n_bins: Number of bins (deciles by default) to divide the population.
            title: Optional title for the chart.
        """
        self._clear_plot()
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(self._default_title)

        if not isinstance(y_true, (pd.Series, np.ndarray)) or \
           not isinstance(y_pred_proba, (pd.Series, np.ndarray)):
            logger.error("y_true and y_pred_proba must be pandas Series or NumPy arrays.")
            self.ax.text(0.5, 0.5, "Invalid input data types.", ha='center', va='center', color='red')
            self.canvas.draw()
            return

        if len(y_true) != len(y_pred_proba):
            logger.error("y_true and y_pred_proba must have the same length.")
            self.ax.text(0.5, 0.5, "Input arrays have different lengths.", ha='center', va='center', color='red')
            self.canvas.draw()
            return
        
        if len(y_true) == 0:
            logger.warning("Input data is empty. Cannot generate lift chart.")
            self.ax.text(0.5, 0.5, "Input data is empty.", ha='center', va='center')
            self.canvas.draw()
            return

        try:
            if isinstance(y_true, pd.Series): y_true_binary = (y_true == positive_class_label).astype(int).values
            else: y_true_binary = (y_true == positive_class_label).astype(int)

            if isinstance(y_pred_proba, pd.Series): y_pred_proba = y_pred_proba.values

            data = pd.DataFrame({'y_true': y_true_binary, 'y_pred_proba': y_pred_proba})
            data = data.sort_values(by='y_pred_proba', ascending=False).reset_index(drop=True)

            data['bin'] = pd.qcut(data.index, n_bins, labels=False, duplicates='drop')
            
            total_positives = data['y_true'].sum()
            total_observations = len(data)
            overall_positive_rate = total_positives / total_observations
            
            logger.info(f"Overall response rate (baseline): {total_positives}/{total_observations} = {overall_positive_rate:.4f} ({overall_positive_rate*100:.1f}%)")
            
            if overall_positive_rate == 0:
                logger.warning("No positive instances in y_true. Lift chart cannot be generated meaningfully.")
                self.ax.text(0.5, 0.5, "No positive instances found.", ha='center', va='center', color='orange')
                self.canvas.draw()
                return

            lift_values = []
            binned_positives = data.groupby('bin')['y_true'].sum()
            binned_counts = data.groupby('bin')['y_true'].count()

            for i in range(n_bins):
                if i in binned_counts.index and binned_counts[i] > 0:
                    actual_positives_in_bin = binned_positives.get(i, 0)
                    total_observations_in_bin = binned_counts[i]
                    bin_response_rate = actual_positives_in_bin / total_observations_in_bin
                    
                    if overall_positive_rate > 0:
                        lift = bin_response_rate / overall_positive_rate
                    else:
                        lift = 0
                    
                    lift_values.append(lift)
                    
                    # Log details for debugging
                    logger.debug(f"Bin {i+1}: {actual_positives_in_bin}/{total_observations_in_bin} = "
                               f"{bin_response_rate:.4f} response rate, lift = {lift:.2f}")
                else:
                    lift_values.append(0)


            bin_labels = [f"{i*100/n_bins:.0f}-{(i+1)*100/n_bins:.0f}%" for i in range(n_bins)]
            
            if len(lift_values) != n_bins: # If qcut dropped bins due to duplicates
                logger.warning(f"Number of bins created ({len(lift_values)}) is less than requested ({n_bins}) due to data distribution. Adjusting plot.")
                actual_n_bins = len(lift_values)
                bin_labels = [f"Bin {i+1}" for i in range(actual_n_bins)] # Simpler labels if bins are irregular
            else:
                actual_n_bins = n_bins

            self.ax.bar(range(actual_n_bins), lift_values, color='skyblue', edgecolor='black', label='Model Lift')
            self.ax.axhline(1, color='grey', linestyle='--', label='Baseline (Random)')

            self.ax.set_xticks(range(actual_n_bins))
            self.ax.set_xticklabels(bin_labels, rotation=45, ha="right")
            self.ax.set_ylabel("Lift")
            self.ax.set_xlabel(f"Population Percentiles (in {actual_n_bins} bins)")
            self.ax.legend()
            self.ax.set_ylim(bottom=0) # Lift should not be negative

        except Exception as e:
            logger.error(f"Error generating lift chart: {e}", exc_info=True)
            self.ax.text(0.5, 0.5, "Error generating chart.", ha='center', va='center', color='red')
        finally:
            self.figure.tight_layout()
            self.canvas.draw()

    def clear_chart(self):
        self._clear_plot()


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    main_win = QMainWindow()
    lift_widget = LiftChartWidget(main_win)
    main_win.setCentralWidget(lift_widget)
    main_win.setWindowTitle("Lift Chart Test")
    main_win.setGeometry(100, 100, 600, 500)
    main_win.show()

    np.random.seed(42)
    n_samples = 1000
    y_true_test = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    y_pred_proba_test = np.clip(
        y_true_test * 0.5 + np.random.normal(0.3, 0.2, n_samples) + 0.1, 0, 1
    )
    

    lift_widget.plot_lift_chart(y_true_test, y_pred_proba_test, positive_class_label=1, n_bins=10)

    sys.exit(app.exec_())