import pandas as pd
import numpy as np
import os
import logging
from supply_chain_optimization import SupplyChainOptimizer
from demand_forecasting import DemandForecaster
from visualization import SupplyChainVisualizer
from scenario_analysis import ScenarioAnalyzer
from risk_analysis import RiskAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_data_path() -> str:
    """Get the absolute path to the data file."""
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (parent of src)
    project_root = os.path.dirname(script_dir)
    # Construct the path to the data file
    data_path = os.path.join(project_root, 'data', 'DataCoSupplyChainDataset.csv')
    
    if not os.path.exists(data_path):
        # Try alternative filename
        alt_data_path = os.path.join(project_root, 'data', 'dataco_smart.csv')
        if os.path.exists(alt_data_path):
            return alt_data_path
        else:
            raise FileNotFoundError(
                f"Data file not found. Please ensure the dataset is in the 'data' directory. "
                f"Tried: {data_path} and {alt_data_path}"
            )
    return data_path

def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare the dataset."""
    try:
        data_path = get_data_path()
        logger.info(f"Loading data from: {data_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'ISO-8859-1', 'latin1']
        for encoding in encodings:
            try:
                data = pd.read_csv(data_path, encoding=encoding)
                logger.info(f"Successfully loaded data with {encoding} encoding")
                
                # Verify required columns exist
                required_columns = ['order date (DateOrders)', 'Sales']
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Convert date column to datetime
                data['order date (DateOrders)'] = pd.to_datetime(data['order date (DateOrders)'])
                
                # Sort by date
                data = data.sort_values('order date (DateOrders)')
                
                # Remove any rows with missing values in required columns
                data = data.dropna(subset=required_columns)
                
                logger.info(f"Data prepared successfully. Shape: {data.shape}")
                return data
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading data with {encoding} encoding: {str(e)}")
                continue
        
        raise ValueError("Failed to load data with any of the attempted encodings")
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def scale_demand(demand: list, production_capacity: list, scale_factor: float = 0.5) -> tuple:
    """
    Scale demand to ensure feasibility.
    
    Args:
        demand: List of demand values
        production_capacity: List of production capacities
        scale_factor: Maximum ratio of demand to capacity (default: 0.5)
    
    Returns:
        Tuple of (scaled_demand, was_scaled)
    """
    total_capacity = sum(production_capacity)
    total_demand = sum(demand)
    
    # First normalize the demand to be proportional to capacity
    capacity_ratio = total_capacity / total_demand
    normalized_demand = [d * capacity_ratio for d in demand]
    
    # Then apply the scale factor
    if total_demand > total_capacity * scale_factor:
        new_scale = (total_capacity * scale_factor) / total_demand
        scaled_demand = [d * new_scale for d in demand]
        logger.warning(f"Demand scaled down to {scale_factor*100}% of total capacity for feasibility")
        logger.info(f"Original demand: {demand}")
        logger.info(f"Normalized demand: {normalized_demand}")
        logger.info(f"Scaled demand: {scaled_demand}")
        return scaled_demand, True
    
    return normalized_demand, False

def main():
    """Main function to run the supply chain optimization."""
    try:
        # Load and prepare data
        logger.info("Loading data...")
        data = load_and_prepare_data()
        logger.info(f"Data prepared successfully. Shape: {data.shape}")
        
        # Initialize demand forecaster
        logger.info("Initializing demand forecaster...")
        forecaster = DemandForecaster(
            data=data,
            date_column='order date (DateOrders)',
            target_column='Sales'
        )
        
        # Fit SARIMA model
        logger.info("Fitting SARIMA model...")
        forecaster.fit()
        
        # Generate forecast
        logger.info("Generating demand forecast...")
        forecast, forecast_intervals = forecaster.forecast_demand(steps=12)
        
        # Calculate seasonal factors
        logger.info("Calculating seasonal factors...")
        seasonal_factors = forecaster.calculate_seasonal_factors()
        
        # Set up optimization parameters
        logger.info("Setting up optimization parameters...")
        
        # Production capacity (units per month)
        production_capacity = [1800, 1800, 1700]  # Total: 5300 units
        
        # Storage capacity (units)
        storage_capacity = [2000, 2000, 2000]  # Same for all factories
        
        # Shipping costs (per unit)
        shipping_costs = {
            (0, 0): 5.0, (0, 1): 6.3, (0, 2): 4.0,  # Factory 0 to Customers
            (1, 0): 6.3, (1, 1): 5.0, (1, 2): 4.0,  # Factory 1 to Customers
            (2, 0): 4.0, (2, 1): 4.0, (2, 2): 5.0   # Factory 2 to Customers
        }
        
        # Storage costs (per unit per month)
        storage_costs = [2.0, 2.0, 2.0]  # Same for all factories
        
        # Production costs (per unit)
        production_costs = [10.0, 10.0, 10.0]  # Same for all factories
        
        # Get forecasted demand and scale it
        demand = forecast[-3:].tolist()  # Last 3 months of forecast
        
        # Calculate total capacity and demand
        total_capacity = sum(production_capacity)
        total_demand = sum(demand)
        
        # Configuration for demand scaling
        max_capacity_utilization = 0.6  # Maximum 60% of total capacity
        min_capacity_utilization = 0.3  # Minimum 30% of total capacity
        
        # Determine if scaling is needed
        if total_demand > total_capacity * max_capacity_utilization:
            # Calculate required scaling factor
            required_scale = (total_capacity * max_capacity_utilization) / total_demand
            logger.warning(
                f"Demand ({total_demand:,.2f}) exceeds {max_capacity_utilization*100:.0f}% of total capacity "
                f"({total_capacity:,.2f}). Scaling down by factor {required_scale:.2f} for feasibility."
            )
            
            # Normalize demand to match capacity proportions
            normalized_demand = [d * total_capacity / total_demand for d in demand]
            
            # Scale down to max_capacity_utilization
            scaled_demand = [d * max_capacity_utilization for d in normalized_demand]
            
            logger.info(f"Original demand: {demand}")
            logger.info(f"Normalized demand: {normalized_demand}")
            logger.info(f"Scaled demand: {scaled_demand}")
            demand = scaled_demand
            
        elif total_demand < total_capacity * min_capacity_utilization:
            # Calculate required scaling factor
            required_scale = (total_capacity * min_capacity_utilization) / total_demand
            logger.warning(
                f"Demand ({total_demand:,.2f}) is below {min_capacity_utilization*100:.0f}% of total capacity "
                f"({total_capacity:,.2f}). Scaling up by factor {required_scale:.2f} for efficiency."
            )
            
            # Normalize demand to match capacity proportions
            normalized_demand = [d * total_capacity / total_demand for d in demand]
            
            # Scale up to min_capacity_utilization
            scaled_demand = [d * min_capacity_utilization for d in normalized_demand]
            
            logger.info(f"Original demand: {demand}")
            logger.info(f"Normalized demand: {normalized_demand}")
            logger.info(f"Scaled demand: {scaled_demand}")
            demand = scaled_demand
            
        else:
            logger.info(
                f"Demand ({total_demand:,.2f}) is within feasible range "
                f"({min_capacity_utilization*100:.0f}% to {max_capacity_utilization*100:.0f}% of capacity)."
            )
        
        logger.info(f"Using demand values: {demand}")
        logger.info(f"Total demand: {sum(demand):,.2f}, Total capacity: {total_capacity:,.2f}")
        
        # Initialize optimizer
        logger.info("Initializing supply chain optimizer...")
        optimizer = SupplyChainOptimizer(
            production_capacity=production_capacity,
            shipping_costs=shipping_costs,
            storage_costs=storage_costs,
            production_costs=production_costs,
            storage_capacity=storage_capacity,
            demand=demand
        )
        
        # Solve optimization problem
        logger.info("Solving optimization problem...")
        solution, total_cost = optimizer.solve()
        
        if solution is None:
            logger.error("Optimization failed: No feasible solution found. Check demand, capacity, and constraints.")
            return
        
        # Create visualizations
        logger.info("Creating visualizations...")
        visualizer = SupplyChainVisualizer(solution, total_cost)
        
        # Create and save visualizations
        visualizer.create_production_plan()
        visualizer.create_shipping_plan()
        visualizer.create_inventory_levels()
        visualizer.create_cost_breakdown()
        
        # Create interactive dashboard
        logger.info("Creating interactive dashboard...")
        visualizer.create_dashboard()
        
        logger.info("Supply chain optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 