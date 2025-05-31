# Advanced Supply Chain Optimization

This project implements an advanced supply chain optimization system leveraging predictive analytics and machine learning. It uses the DataCo SMART dataset to optimize supply chain operations, including demand forecasting, inventory management, and risk analysis.

## Features

- **Demand Forecasting**
  - SARIMA-based time series forecasting
  - Seasonal pattern analysis
  - Demand volatility handling

- **Supply Chain Optimization**
  - Multi-factory production planning
  - Multi-customer distribution optimization
  - Inventory level optimization
  - Cost minimization (production, shipping, storage)

- **Risk Analysis**
  - Supply chain risk identification
  - Risk scoring and prioritization
  - Mitigation strategy recommendations
  - Risk matrix visualization

- **Scenario Analysis**
  - What-if analysis for different scenarios
  - Demand variation analysis
  - Cost change impact assessment
  - Capacity reduction analysis

## Project Structure

```
├── src/
│   ├── main.py                 # Main execution script
│   ├── demand_forecasting.py   # Demand forecasting module
│   ├── supply_chain_optimization.py  # Optimization module
│   ├── risk_analysis.py        # Risk analysis module
│   ├── scenario_analysis.py    # Scenario analysis module
│   └── visualization.py        # Visualization module
├── data/
│   └── dataco_smart.csv       # Dataset file
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd supply-chain-optimization
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main optimization script:
```bash
python src/main.py
```

2. The script will:
   - Load and preprocess the data
   - Generate demand forecasts
   - Optimize the supply chain
   - Perform risk analysis
   - Generate visualizations

## Configuration

The optimization parameters can be adjusted in `src/main.py`:

- Production costs
- Shipping costs
- Storage costs
- Production capacity
- Storage capacity
- Minimum inventory levels
- Safety stock factors

### Parameter Tuning Guide

To handle infeasible optimization problems, you can adjust these key parameters:

1. **Demand Scaling**
   ```python
   # Scale demand to 70% of total capacity for feasibility
   total_capacity = sum(production_capacity)
   total_demand = sum(demand)
   if total_demand > total_capacity * 0.7:
       scale_factor = (total_capacity * 0.7) / total_demand
       demand = [d * scale_factor for d in demand]
   ```

2. **Constraint Parameters**
   ```python
   optimizer = SupplyChainOptimizer(
       min_inventory_level=0.01,      # Minimum inventory (1% of capacity)
       max_factory_utilization=0.95,   # Maximum factory utilization (95%)
       safety_stock_factor=0.05       # Safety stock (5% of demand)
   )
   ```

3. **Recommended Parameter Ranges**
   - `min_inventory_level`: 0.01 to 0.05
   - `max_factory_utilization`: 0.90 to 0.95
   - `safety_stock_factor`: 0.05 to 0.10
   - Demand scaling: 0.70 to 0.80 of total capacity

### Troubleshooting Infeasible Problems

If you encounter infeasible optimization problems:

1. **Check Demand vs. Capacity**
   - Ensure total demand is less than 70% of total capacity
   - Scale demand if necessary
   - Verify capacity constraints are realistic

2. **Adjust Constraints**
   - Reduce minimum inventory levels
   - Increase maximum factory utilization
   - Lower safety stock requirements
   - Relax production balance constraints

3. **Monitor Logs**
   - Check warning messages about demand scaling
   - Review constraint violations
   - Verify parameter values

4. **Common Solutions**
   - Scale down demand
   - Increase capacity
   - Relax minimum production requirements
   - Adjust inventory constraints
   - Modify safety stock levels

## Output

The system generates:
- Optimization results
- Cost breakdowns
- Production plans
- Inventory levels
- Risk analysis reports
- Scenario analysis summaries
- Interactive visualizations

## Dependencies

- Python 3.8+
- pandas
- numpy
- pulp
- matplotlib
- seaborn
- statsmodels
- scikit-learn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DataCo SMART dataset
- PuLP optimization library
- Statsmodels for time series analysis