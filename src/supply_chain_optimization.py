# src/supply_chain_optimization.py

import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupplyChainOptimizer:
    def __init__(self, 
                 production_costs: List[float],
                 shipping_costs: List[float],
                 storage_costs: List[float],
                 production_capacity: List[float],
                 storage_capacity: List[float],
                 demand: List[float],
                 seasonal_factors: Dict[str, float] = None,
                 min_inventory_level: float = 0.1,  # Minimum inventory level as percentage of demand
                 max_factory_utilization: float = 0.8,  # Maximum factory utilization percentage
                 safety_stock_factor: float = 0.2):  # Safety stock as percentage of demand
        """
        Initialize the Supply Chain Optimizer.
        
        Args:
            production_costs: List of production costs per unit for each factory
            shipping_costs: List of shipping costs per unit for each customer
            storage_costs: List of storage costs per unit for each factory
            production_capacity: List of production capacity for each factory
            storage_capacity: List of storage capacity for each factory
            demand: List of demand for each customer
            seasonal_factors: Dictionary of seasonal adjustment factors
            min_inventory_level: Minimum inventory level as percentage of demand
            max_factory_utilization: Maximum factory utilization percentage
            safety_stock_factor: Safety stock as percentage of demand
        """
        self.production_costs = production_costs
        self.shipping_costs = shipping_costs
        self.storage_costs = storage_costs
        self.production_capacity = production_capacity
        self.storage_capacity = storage_capacity
        self.demand = demand
        self.seasonal_factors = seasonal_factors or {}
        self.min_inventory_level = min_inventory_level
        self.max_factory_utilization = max_factory_utilization
        self.safety_stock_factor = safety_stock_factor
        
        # Initialize the optimization problem
        self.prob = None
        self.x = None  # Production variables
        self.y = None  # Shipping variables
        self.inventory = None  # Inventory variables
        
    def create_optimization_problem(self):
        """Create the linear programming optimization problem."""
        # Create the optimization problem
        self.prob = pulp.LpProblem("SupplyChainOptimization", pulp.LpMinimize)
        
        # Create decision variables
        num_factories = len(self.production_costs)
        num_customers = len(self.demand)
        
        # Production variables (x[i] = amount produced at factory i)
        self.x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Continuous') 
                 for i in range(num_factories)]
        
        # Shipping variables (y[i,j] = amount shipped from factory i to customer j)
        self.y = [[pulp.LpVariable(f"y_{i}_{j}", lowBound=0, cat='Continuous')
                  for j in range(num_customers)]
                 for i in range(num_factories)]
        
        # Inventory variables (inventory[i] = amount stored at factory i)
        self.inventory = [pulp.LpVariable(f"inventory_{i}", lowBound=0, cat='Continuous')
                         for i in range(num_factories)]
        
        # Objective Function: Minimize total cost
        # Production cost + Shipping cost + Storage cost
        production_cost = pulp.lpSum([self.production_costs[i] * self.x[i] 
                                    for i in range(num_factories)])
        
        shipping_cost = pulp.lpSum([self.shipping_costs[i * num_customers + j] * self.y[i][j]
                                  for i in range(num_factories)
                                  for j in range(num_customers)])
        
        storage_cost = pulp.lpSum([self.storage_costs[i] * self.inventory[i]
                                 for i in range(num_factories)])
        
        self.prob += production_cost + shipping_cost + storage_cost
        
        # Constraints
        # 1. Production capacity constraints with utilization limit
        for i in range(num_factories):
            self.prob += self.x[i] <= self.production_capacity[i] * self.max_factory_utilization, \
                        f"Production_Capacity_{i}"
        
        # 2. Storage capacity constraints
        for i in range(num_factories):
            self.prob += self.inventory[i] <= self.storage_capacity[i], \
                        f"Storage_Capacity_{i}"
        
        # 3. Demand fulfillment constraints with seasonal factors
        for j in range(num_customers):
            demand = self.demand[j]
            if self.seasonal_factors:
                # Apply seasonal adjustment if available
                month = list(self.seasonal_factors.keys())[j % len(self.seasonal_factors)]
                demand *= self.seasonal_factors[month]
            
            # Sum of shipments from all factories to customer j must equal demand
            self.prob += pulp.lpSum([self.y[i][j] for i in range(num_factories)]) == demand, \
                        f"Demand_Fulfillment_{j}"
        
        # 4. Inventory balance constraints
        for i in range(num_factories):
            # Production = Shipments + Current Inventory
            self.prob += self.x[i] == pulp.lpSum([self.y[i][j] for j in range(num_customers)]) + self.inventory[i], \
                        f"Inventory_Balance_{i}"
        
        # 5. Minimum production constraints (to ensure factories are utilized)
        for i in range(num_factories):
            min_production = 0.05 * self.production_capacity[i]  # Reduced to 5% minimum utilization
            self.prob += self.x[i] >= min_production, f"Min_Production_{i}"
        
        # 6. Minimum inventory level constraints (relaxed)
        total_demand = sum(self.demand)
        min_inventory = total_demand * self.min_inventory_level / num_factories
        for i in range(num_factories):
            self.prob += self.inventory[i] >= min_inventory, \
                        f"Min_Inventory_{i}"
        
        # 7. Safety stock constraints (relaxed)
        safety_stock = total_demand * self.safety_stock_factor / num_factories
        for i in range(num_factories):
            self.prob += self.inventory[i] >= safety_stock, \
                        f"Safety_Stock_{i}"
        
        # 8. Production balance constraints (relaxed)
        avg_production = pulp.lpSum(self.x) / num_factories
        for i in range(num_factories):
            # Allow deviation of up to 50% from average (increased from 30%)
            self.prob += self.x[i] <= avg_production * 1.5, f"Max_Production_Deviation_{i}"
            self.prob += self.x[i] >= avg_production * 0.5, f"Min_Production_Deviation_{i}"
    
    def solve(self) -> Tuple[Dict, float]:
        """
        Solve the supply chain optimization problem.
        
        Returns:
            Tuple of (solution dictionary, total cost)
        """
        try:
            # Create optimization problem
            self.prob = pulp.LpProblem("Supply_Chain_Optimization", pulp.LpMinimize)
            
            # Create variables
            num_factories = len(self.production_costs)
            num_customers = len(self.demand)
            num_periods = len(self.seasonal_factors) if self.seasonal_factors else 1
            
            production = pulp.LpVariable.dicts(
                "production",
                ((i, t) for i in range(num_factories) for t in range(num_periods)),
                lowBound=0
            )
            
            shipping = pulp.LpVariable.dicts(
                "shipping",
                ((i, j, t) for i in range(num_factories) 
                 for j in range(num_customers) for t in range(num_periods)),
                lowBound=0
            )
            
            inventory = pulp.LpVariable.dicts(
                "inventory",
                ((i, t) for i in range(num_factories) for t in range(num_periods)),
                lowBound=0
            )
            
            # Objective function
            self.prob += (
                pulp.lpSum(production[i, t] * self.production_costs[i] 
                          for i in range(num_factories) for t in range(num_periods)) +
                pulp.lpSum(shipping[i, j, t] * self.shipping_costs[i * num_customers + j] 
                          for i in range(num_factories) 
                          for j in range(num_customers) for t in range(num_periods)) +
                pulp.lpSum(inventory[i, t] * self.storage_costs[i] 
                          for i in range(num_factories) for t in range(num_periods))
            )
            
            # Constraints
            # Production capacity
            for i in range(num_factories):
                for t in range(num_periods):
                    self.prob += production[i, t] <= self.production_capacity[i]
            
            # Storage capacity
            for i in range(num_factories):
                for t in range(num_periods):
                    self.prob += inventory[i, t] <= self.storage_capacity[i]
            
            # Demand satisfaction
            for j in range(num_customers):
                for t in range(num_periods):
                    self.prob += (
                        pulp.lpSum(shipping[i, j, t] for i in range(num_factories)) == 
                        self.demand[j]
                    )
            
            # Inventory balance
            for i in range(num_factories):
                for t in range(num_periods):
                    if t == 0:
                        self.prob += (
                            inventory[i, t] == 
                            production[i, t] - pulp.lpSum(shipping[i, j, t] 
                                                         for j in range(num_customers))
                        )
                    else:
                        self.prob += (
                            inventory[i, t] == 
                            inventory[i, t-1] + production[i, t] - 
                            pulp.lpSum(shipping[i, j, t] for j in range(num_customers))
                        )
            
            # Solve the problem
            status = self.prob.solve(pulp.PULP_CBC_CMD(msg=True))
            
            if status != pulp.LpStatusOptimal:
                logger.error(f"Optimization failed with status: {pulp.LpStatus[status]}")
                return None, 0.0
            
            # Extract solution
            solution = {
                'production': {
                    (i, t): production[i, t].value()
                    for i in range(num_factories)
                    for t in range(num_periods)
                },
                'shipping': {
                    (i, j, t): shipping[i, j, t].value()
                    for i in range(num_factories)
                    for j in range(num_customers)
                    for t in range(num_periods)
                },
                'inventory': {
                    (i, t): inventory[i, t].value()
                    for i in range(num_factories)
                    for t in range(num_periods)
                }
            }
            
            total_cost = pulp.value(self.prob.objective)
            
            logger.info("Optimization completed successfully")
            logger.info(f"Total cost: {total_cost:,.2f}")
            
            # Log production levels
            logger.info("Production levels:")
            for i in range(num_factories):
                total_prod = sum(solution['production'][i, t] for t in range(num_periods))
                logger.info(f"Factory {i}: {total_prod:,.2f} units")
            
            return solution, total_cost
            
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            return None, 0.0
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """
        Calculate the cost breakdown from the optimization results.
        
        Returns:
            Dictionary with cost breakdown
        """
        try:
            if not hasattr(self, 'prob') or self.prob is None:
                logger.error("No optimization problem has been solved yet")
                return None
                
            num_factories = len(self.production_costs)
            num_customers = len(self.demand)
            
            # Calculate production costs
            production_cost = 0
            for i in range(num_factories):
                var_name = f'production_{i}'
                if var_name in self.prob.variables():
                    production_cost += self.production_costs[i] * self.prob.variables()[var_name].value()
            
            # Calculate shipping costs
            shipping_cost = 0
            for i in range(num_factories):
                for j in range(num_customers):
                    var_name = f'shipping_{i}_{j}'
                    if var_name in self.prob.variables():
                        shipping_cost += self.shipping_costs[i * num_customers + j] * self.prob.variables()[var_name].value()
            
            # Calculate storage costs
            storage_cost = 0
            for i in range(num_factories):
                var_name = f'inventory_{i}'
                if var_name in self.prob.variables():
                    storage_cost += self.storage_costs[i] * self.prob.variables()[var_name].value()
            
            total_cost = production_cost + shipping_cost + storage_cost
            
            return {
                'Production Cost': round(production_cost, 2),
                'Shipping Cost': round(shipping_cost, 2),
                'Storage Cost': round(storage_cost, 2),
                'Total Cost': round(total_cost, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating cost breakdown: {str(e)}")
            return None

# Simulating seasonal demand: Increase demand during months like December (holidays) or sales periods.
seasonal_demand_factor = {
    'January': 1.05,   # 5% increase in January
    'February': 1.0,   # Regular demand
    'March': 1.0,      # Regular demand
    'April': 0.95,     # Slight decrease
    'May': 1.0,        # Regular demand
    'June': 1.1,       # 10% increase during summer
    'July': 1.05,      # 5% increase
    'August': 1.0,     # Regular demand
    'September': 0.9,  # Decrease in demand
    'October': 1.05,   # Increased demand during sales periods
    'November': 1.1,   # Increased demand before holidays
    'December': 1.15   # Peak demand during holidays
}

# Sample data for production costs, shipping costs, and demand
production_costs = [10, 15, 20]  # Cost per unit at each factory
base_shipping_costs = [5, 7, 4]  # Base cost per unit for each shipping route
demand = [200, 300, 250]  # Base forecasted demand for each customer
storage_costs = [2, 3, 1]  # Cost per unit stored in each factory's warehouse (storage costs)
lead_times = [1, 2, 3]  # Lead time (in months) for each factory to produce a unit

# Define production capacity for each factory
production_capacity = [1000, 1500, 1200]
min_production = [100, 100, 100]  # Minimum units to be produced at each factory

# Relaxed storage capacity (increase from previous values)
storage_capacity = [1200, 1500, 1200]  # Increased storage capacity for each factory

# Define the Linear Programming problem (minimization)
prob = pulp.LpProblem("SupplyChainOptimization", pulp.LpMinimize)

# Decision variables (amount to produce at each factory)
x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Continuous') for i in range(len(production_costs))]

# Decision variables (amount to ship to each customer)
y = [pulp.LpVariable(f"y_{i}", lowBound=0, cat='Continuous') for i in range(len(base_shipping_costs))]

# Decision variables (inventory at each factory to incur storage costs)
inventory = [pulp.LpVariable(f"inventory_{i}", lowBound=0, cat='Continuous') for i in range(len(storage_costs))]

# Objective: Minimize the total cost (production + shipping + storage)
prob += pulp.lpSum([production_costs[i] * x[i] for i in range(len(x))]) + \
         pulp.lpSum([base_shipping_costs[i] * y[i] for i in range(len(y))]) + \
         pulp.lpSum([storage_costs[i] * inventory[i] for i in range(len(inventory))]), "Total Cost"

# Constraints:
# 1. Meet the seasonal demand for each customer, adjust based on seasonal factors
for month, factor in seasonal_demand_factor.items():
    # Adjust demand for each month based on the seasonal demand factor
    seasonal_demand = [d * factor for d in demand]
    prob += y[0] == seasonal_demand[0], f"Seasonal Demand for Customer 1 in {month}"
    prob += y[1] == seasonal_demand[1], f"Seasonal Demand for Customer 2 in {month}"
    prob += y[2] == seasonal_demand[2], f"Seasonal Demand for Customer 3 in {month}"

# 2. Do not exceed production capacity at each factory
for i in range(len(production_capacity)):
    prob += x[i] <= production_capacity[i], f"Production Capacity at Factory {i+1}"

# 3. Minimum production at each factory (relaxed to allow more flexibility)
for i in range(len(min_production)):
    prob += x[i] >= min_production[i], f"Minimum Production at Factory {i+1}"

# 4. Ensure that inventory at each factory equals produced units minus shipped units
for i in range(len(production_capacity)):
    prob += inventory[i] == x[i] - sum([y[j] for j in range(len(demand)) if j == i]), f"Inventory at Factory {i+1}"

# 5. Relaxed storage constraints (increase storage capacity)
for i in range(len(storage_capacity)):
    prob += inventory[i] <= storage_capacity[i], f"Storage Capacity at Factory {i+1}"

# 6. Production lead times (reduce the impact of lead time for flexibility)
for i in range(len(lead_times)):
    prob += x[i] >= demand[i] * 0.9, f"Adjusted Lead time constraint at Factory {i+1}"  # Relaxed to 90%

# Solve the problem
prob.solve()

# Bulk shipping discount logic after the optimization is solved
def bulk_shipping_discount(shipped_units, base_cost):
    if shipped_units > 500:
        return base_cost * 0.85  # 15% discount for large orders
    elif shipped_units > 300:
        return base_cost * 0.90  # 10% discount for medium orders
    else:
        return base_cost  # No discount for small orders

# Display the results and apply bulk shipping discounts
print("Optimization Results:")
for i in range(len(x)):
    print(f"Produce {x[i].varValue} units at factory {i+1}")
for i in range(len(y)):
    discounted_shipping_cost = bulk_shipping_discount(y[i].varValue, base_shipping_costs[i])
    print(f"Ship {y[i].varValue} units to customer {i+1} at discounted cost: {discounted_shipping_cost * y[i].varValue}")
for i in range(len(inventory)):
    print(f"Store {inventory[i].varValue} units at factory {i+1}")

print("\nTotal Cost:", pulp.value(prob.objective))
