import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from supply_chain_optimization import SupplyChainOptimizer
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Types of scenarios to analyze."""
    DEMAND_INCREASE = "demand_increase"
    DEMAND_DECREASE = "demand_decrease"
    COST_INCREASE = "cost_increase"
    COST_DECREASE = "cost_decrease"
    CAPACITY_REDUCTION = "capacity_reduction"
    SUPPLY_DISRUPTION = "supply_disruption"
    SEASONAL_PEAK = "seasonal_peak"
    SEASONAL_LOW = "seasonal_low"

@dataclass
class Scenario:
    """Class to represent a supply chain scenario."""
    name: str
    type: ScenarioType
    parameters: Dict[str, float]
    probability: float = 1.0
    description: str = ""

class ScenarioAnalyzer:
    def __init__(self, base_optimizer: SupplyChainOptimizer):
        """
        Initialize the Scenario Analyzer.
        
        Args:
            base_optimizer: Base SupplyChainOptimizer instance
        """
        self.base_optimizer = base_optimizer
        self.scenarios = []
        self.results = {}
        
    def add_scenario(self, scenario: Scenario) -> None:
        """
        Add a scenario to analyze.
        
        Args:
            scenario: Scenario object to add
        """
        self.scenarios.append(scenario)
        
    def create_demand_scenarios(self,
                              base_demand: List[float],
                              increase_percentages: List[float] = [10, 20, 30],
                              decrease_percentages: List[float] = [10, 20, 30]) -> None:
        """
        Create scenarios for demand increase and decrease.
        
        Args:
            base_demand: Base demand values
            increase_percentages: List of demand increase percentages
            decrease_percentages: List of demand decrease percentages
        """
        # Create demand increase scenarios
        for pct in increase_percentages:
            scenario = Scenario(
                name=f"Demand Increase {pct}%",
                type=ScenarioType.DEMAND_INCREASE,
                parameters={'demand': [d * (1 + pct/100) for d in base_demand]},
                probability=0.2,
                description=f"Scenario with {pct}% increase in demand"
            )
            self.add_scenario(scenario)
        
        # Create demand decrease scenarios
        for pct in decrease_percentages:
            scenario = Scenario(
                name=f"Demand Decrease {pct}%",
                type=ScenarioType.DEMAND_DECREASE,
                parameters={'demand': [d * (1 - pct/100) for d in base_demand]},
                probability=0.2,
                description=f"Scenario with {pct}% decrease in demand"
            )
            self.add_scenario(scenario)
    
    def create_cost_scenarios(self,
                            base_costs: Dict[str, List[float]],
                            increase_percentages: List[float] = [10, 20, 30],
                            decrease_percentages: List[float] = [10, 20, 30]) -> None:
        """
        Create scenarios for cost increase and decrease.
        
        Args:
            base_costs: Dictionary of base costs
            increase_percentages: List of cost increase percentages
            decrease_percentages: List of cost decrease percentages
        """
        # Create cost increase scenarios
        for cost_type, costs in base_costs.items():
            for pct in increase_percentages:
                scenario = Scenario(
                    name=f"{cost_type.title()} Increase {pct}%",
                    type=ScenarioType.COST_INCREASE,
                    parameters={cost_type: [c * (1 + pct/100) for c in costs]},
                    probability=0.15,
                    description=f"Scenario with {pct}% increase in {cost_type}"
                )
                self.add_scenario(scenario)
        
        # Create cost decrease scenarios
        for cost_type, costs in base_costs.items():
            for pct in decrease_percentages:
                scenario = Scenario(
                    name=f"{cost_type.title()} Decrease {pct}%",
                    type=ScenarioType.COST_DECREASE,
                    parameters={cost_type: [c * (1 - pct/100) for c in costs]},
                    probability=0.15,
                    description=f"Scenario with {pct}% decrease in {cost_type}"
                )
                self.add_scenario(scenario)
    
    def create_capacity_scenarios(self,
                                base_capacity: List[float],
                                reduction_percentages: List[float] = [20, 40, 60]) -> None:
        """
        Create scenarios for capacity reduction.
        
        Args:
            base_capacity: Base capacity values
            reduction_percentages: List of capacity reduction percentages
        """
        for pct in reduction_percentages:
            scenario = Scenario(
                name=f"Capacity Reduction {pct}%",
                type=ScenarioType.CAPACITY_REDUCTION,
                parameters={'production_capacity': [c * (1 - pct/100) for c in base_capacity]},
                probability=0.1,
                description=f"Scenario with {pct}% reduction in production capacity"
            )
            self.add_scenario(scenario)
    
    def create_seasonal_scenarios(self,
                                base_demand: List[float],
                                peak_factor: float = 1.5,
                                low_factor: float = 0.7) -> None:
        """
        Create scenarios for seasonal variations.
        
        Args:
            base_demand: Base demand values
            peak_factor: Factor for peak season
            low_factor: Factor for low season
        """
        # Create peak season scenario
        scenario = Scenario(
            name="Seasonal Peak",
            type=ScenarioType.SEASONAL_PEAK,
            parameters={'demand': [d * peak_factor for d in base_demand]},
            probability=0.25,
            description="Scenario for peak season demand"
        )
        self.add_scenario(scenario)
        
        # Create low season scenario
        scenario = Scenario(
            name="Seasonal Low",
            type=ScenarioType.SEASONAL_LOW,
            parameters={'demand': [d * low_factor for d in base_demand]},
            probability=0.25,
            description="Scenario for low season demand"
        )
        self.add_scenario(scenario)
    
    def analyze_scenarios(self) -> pd.DataFrame:
        """
        Analyze all scenarios and return results.
        
        Returns:
            DataFrame with scenario analysis results
        """
        results = []
        
        for scenario in self.scenarios:
            # Create optimizer with scenario parameters
            optimizer = self._create_scenario_optimizer(scenario)
            
            # Solve the optimization problem
            solution, cost = optimizer.solve()
            
            if solution is not None:
                # Calculate metrics
                metrics = self._calculate_scenario_metrics(solution, cost)
                
                # Add results
                result = {
                    'scenario_name': scenario.name,
                    'scenario_type': scenario.type.value,
                    'probability': scenario.probability,
                    'description': scenario.description,
                    'total_cost': cost,
                    **metrics
                }
                results.append(result)
        
        return pd.DataFrame(results)
    
    def _create_scenario_optimizer(self, scenario: Scenario) -> SupplyChainOptimizer:
        """Create an optimizer with scenario parameters."""
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
        
        # Update with scenario parameters
        params.update(scenario.parameters)
        
        return SupplyChainOptimizer(**params)
    
    def _calculate_scenario_metrics(self,
                                  solution: Dict,
                                  cost: float) -> Dict[str, float]:
        """Calculate metrics for a scenario."""
        return {
            'total_production': sum(solution['production'].values()),
            'total_inventory': sum(solution['inventory'].values()),
            'total_shipping': sum(solution['shipping'].values()),
            'avg_production_per_factory': np.mean(list(solution['production'].values())),
            'avg_inventory_per_factory': np.mean(list(solution['inventory'].values())),
            'production_utilization': sum(solution['production'].values()) / 
                                    sum(self.base_optimizer.production_capacity),
            'inventory_utilization': sum(solution['inventory'].values()) / 
                                   sum(self.base_optimizer.storage_capacity)
        }
    
    def get_scenario_summary(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary of scenario analysis results.
        
        Args:
            results: DataFrame with scenario analysis results
            
        Returns:
            DataFrame with scenario summary
        """
        summary = results.groupby('scenario_type').agg({
            'total_cost': ['mean', 'std', 'min', 'max'],
            'total_production': ['mean', 'std'],
            'total_inventory': ['mean', 'std'],
            'production_utilization': ['mean', 'std'],
            'inventory_utilization': ['mean', 'std']
        }).round(2)
        
        return summary 