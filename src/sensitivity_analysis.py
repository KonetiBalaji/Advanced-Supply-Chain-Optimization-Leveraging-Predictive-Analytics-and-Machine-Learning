import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from supply_chain_optimization import SupplyChainOptimizer
import logging

logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    def __init__(self, base_optimizer: SupplyChainOptimizer):
        """
        Initialize the Sensitivity Analyzer.
        
        Args:
            base_optimizer: Base SupplyChainOptimizer instance with initial parameters
        """
        self.base_optimizer = base_optimizer
        self.results = {}
        
    def analyze_parameter_sensitivity(self,
                                    parameter_name: str,
                                    parameter_values: List[float],
                                    metric: str = 'total_cost') -> pd.DataFrame:
        """
        Analyze how changes in a parameter affect the optimization results.
        
        Args:
            parameter_name: Name of the parameter to analyze
            parameter_values: List of values to test
            metric: Metric to track ('total_cost', 'production', 'inventory', etc.)
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        results = []
        
        for value in parameter_values:
            # Create a copy of the optimizer with modified parameter
            optimizer = self._create_modified_optimizer(parameter_name, value)
            
            # Solve the optimization problem
            solution, cost = optimizer.solve()
            
            if solution is not None:
                # Calculate the metric value
                metric_value = self._calculate_metric(solution, cost, metric)
                
                results.append({
                    'parameter_value': value,
                    'metric_value': metric_value
                })
        
        return pd.DataFrame(results)
    
    def analyze_multiple_parameters(self,
                                  parameters: Dict[str, List[float]],
                                  metric: str = 'total_cost') -> pd.DataFrame:
        """
        Analyze how changes in multiple parameters affect the results.
        
        Args:
            parameters: Dictionary of parameter names and their values to test
            metric: Metric to track
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        results = []
        
        # Generate all combinations of parameter values
        param_combinations = self._generate_parameter_combinations(parameters)
        
        for combination in param_combinations:
            # Create optimizer with current parameter combination
            optimizer = self._create_modified_optimizer_from_dict(combination)
            
            # Solve the optimization problem
            solution, cost = optimizer.solve()
            
            if solution is not None:
                # Calculate the metric value
                metric_value = self._calculate_metric(solution, cost, metric)
                
                # Add results
                result = combination.copy()
                result['metric_value'] = metric_value
                results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_sensitivity_analysis(self,
                                results: pd.DataFrame,
                                parameter_name: str,
                                metric: str,
                                title: str = None) -> None:
        """
        Plot the results of a sensitivity analysis.
        
        Args:
            results: DataFrame with sensitivity analysis results
            parameter_name: Name of the parameter that was varied
            metric: Name of the metric that was tracked
            title: Optional title for the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot the results
        plt.plot(results['parameter_value'], results['metric_value'], 'b-o')
        
        # Add labels and title
        plt.xlabel(parameter_name.replace('_', ' ').title())
        plt.ylabel(metric.replace('_', ' ').title())
        if title:
            plt.title(title)
        else:
            plt.title(f'Sensitivity Analysis: {parameter_name} vs {metric}')
        
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_heatmap(self,
                    results: pd.DataFrame,
                    x_param: str,
                    y_param: str,
                    metric: str,
                    title: str = None) -> None:
        """
        Create a heatmap of the sensitivity analysis results.
        
        Args:
            results: DataFrame with sensitivity analysis results
            x_param: Parameter to plot on x-axis
            y_param: Parameter to plot on y-axis
            metric: Metric to plot as color
            title: Optional title for the plot
        """
        # Pivot the data for the heatmap
        heatmap_data = results.pivot_table(
            values=metric,
            index=y_param,
            columns=x_param
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'Sensitivity Analysis: {x_param} vs {y_param}')
        
        plt.tight_layout()
        plt.show()
    
    def _create_modified_optimizer(self,
                                 parameter_name: str,
                                 parameter_value: float) -> SupplyChainOptimizer:
        """Create a modified optimizer with the specified parameter value."""
        # Get the base parameters
        params = {
            'production_costs': self.base_optimizer.production_costs,
            'shipping_costs': self.base_optimizer.shipping_costs,
            'storage_costs': self.base_optimizer.storage_costs,
            'production_capacity': self.base_optimizer.production_capacity,
            'storage_capacity': self.base_optimizer.storage_capacity,
            'demand': self.base_optimizer.demand,
            'seasonal_factors': self.base_optimizer.seasonal_factors,
            'min_inventory_level': self.base_optimizer.min_inventory_level,
            'max_factory_utilization': self.base_optimizer.max_factory_utilization,
            'safety_stock_factor': self.base_optimizer.safety_stock_factor
        }
        
        # Modify the specified parameter
        params[parameter_name] = parameter_value
        
        # Create and return the modified optimizer
        return SupplyChainOptimizer(**params)
    
    def _create_modified_optimizer_from_dict(self,
                                           parameters: Dict[str, float]) -> SupplyChainOptimizer:
        """Create a modified optimizer with multiple parameter values."""
        # Get the base parameters
        params = {
            'production_costs': self.base_optimizer.production_costs,
            'shipping_costs': self.base_optimizer.shipping_costs,
            'storage_costs': self.base_optimizer.storage_costs,
            'production_capacity': self.base_optimizer.production_capacity,
            'storage_capacity': self.base_optimizer.storage_capacity,
            'demand': self.base_optimizer.demand,
            'seasonal_factors': self.base_optimizer.seasonal_factors,
            'min_inventory_level': self.base_optimizer.min_inventory_level,
            'max_factory_utilization': self.base_optimizer.max_factory_utilization,
            'safety_stock_factor': self.base_optimizer.safety_stock_factor
        }
        
        # Update with the new parameter values
        params.update(parameters)
        
        # Create and return the modified optimizer
        return SupplyChainOptimizer(**params)
    
    def _calculate_metric(self,
                         solution: Dict,
                         cost: float,
                         metric: str) -> float:
        """Calculate the specified metric from the solution."""
        if metric == 'total_cost':
            return cost
        elif metric == 'production':
            return sum(solution['production'].values())
        elif metric == 'inventory':
            return sum(solution['inventory'].values())
        elif metric == 'shipping':
            return sum(solution['shipping'].values())
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _generate_parameter_combinations(self,
                                       parameters: Dict[str, List[float]]) -> List[Dict[str, float]]:
        """Generate all combinations of parameter values."""
        import itertools
        
        # Get the parameter names and their values
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        return [dict(zip(param_names, combo)) for combo in combinations] 