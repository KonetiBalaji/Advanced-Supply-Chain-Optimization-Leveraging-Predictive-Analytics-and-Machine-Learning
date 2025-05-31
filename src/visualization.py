import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List
import os

logger = logging.getLogger(__name__)

class SupplyChainVisualizer:
    def __init__(self, solution: Dict, total_cost: float):
        """
        Initialize the visualizer with optimization results.
        
        Args:
            solution: Dictionary containing the optimization solution
            total_cost: Total cost of the solution
        """
        self.solution = solution
        self.total_cost = total_cost
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plot style
        plt.style.use('seaborn-v0_8')
        
    def create_production_plan(self):
        """Create visualization of production plan."""
        try:
            # Extract production data
            production_data = self.solution['production']
            
            # Create DataFrame for plotting
            df = pd.DataFrame([
                {'Factory': f'Factory {i}', 'Period': t, 'Production': prod}
                for (i, t), prod in production_data.items()
            ])
            
            # Create plot
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x='Period', y='Production', hue='Factory')
            plt.title('Production Plan by Factory')
            plt.xlabel('Time Period')
            plt.ylabel('Production (units)')
            plt.legend(title='Factory')
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'production_plan.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating production plan: {str(e)}")
            raise
    
    def create_shipping_plan(self):
        """Create visualization of shipping plan."""
        try:
            # Extract shipping data
            shipping_data = self.solution['shipping']
            
            # Create DataFrame for plotting
            df = pd.DataFrame([
                {
                    'From': f'Factory {i}',
                    'To': f'Customer {j}',
                    'Period': t,
                    'Units': units
                }
                for (i, j, t), units in shipping_data.items()
            ])
            
            # Create plot
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x='Period', y='Units', hue='From')
            plt.title('Shipping Plan by Factory')
            plt.xlabel('Time Period')
            plt.ylabel('Units Shipped')
            plt.legend(title='Factory')
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'shipping_plan.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating shipping plan: {str(e)}")
            raise
    
    def create_inventory_levels(self):
        """Create visualization of inventory levels."""
        try:
            # Extract inventory data
            inventory_data = self.solution['inventory']
            
            # Create DataFrame for plotting
            df = pd.DataFrame([
                {'Factory': f'Factory {i}', 'Period': t, 'Inventory': inv}
                for (i, t), inv in inventory_data.items()
            ])
            
            # Create plot
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x='Period', y='Inventory', hue='Factory')
            plt.title('Inventory Levels by Factory')
            plt.xlabel('Time Period')
            plt.ylabel('Inventory (units)')
            plt.legend(title='Factory')
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'inventory_levels.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating inventory levels: {str(e)}")
            raise
    
    def create_cost_breakdown(self):
        """Create visualization of cost breakdown."""
        try:
            # Calculate cost components (just sum the values for now)
            production_cost = sum(
                self.solution['production'].get((i, t), 0)
                for i in range(3) for t in range(3)
            )
            
            shipping_cost = sum(
                self.solution['shipping'].get((i, j, t), 0)
                for i in range(3) for j in range(3) for t in range(3)
            )
            
            storage_cost = sum(
                self.solution['inventory'].get((i, t), 0)
                for i in range(3) for t in range(3)
            )
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'Cost Type': ['Production', 'Shipping', 'Storage'],
                'Cost': [production_cost, shipping_cost, storage_cost]
            })
            
            # Create plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='Cost Type', y='Cost')
            plt.title('Cost Breakdown')
            plt.xlabel('Cost Type')
            plt.ylabel('Cost ($)')
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'cost_breakdown.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating cost breakdown: {str(e)}")
            raise
    
    def create_dashboard(self):
        """Create interactive dashboard using Plotly."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Production Plan',
                    'Shipping Plan',
                    'Inventory Levels',
                    'Cost Breakdown'
                )
            )
            
            # Add production plan
            production_data = self.solution['production']
            df_prod = pd.DataFrame([
                {'Factory': f'Factory {i}', 'Period': t, 'Production': prod}
                for (i, t), prod in production_data.items()
            ])
            
            for factory in df_prod['Factory'].unique():
                df_factory = df_prod[df_prod['Factory'] == factory]
                fig.add_trace(
                    go.Bar(
                        x=df_factory['Period'],
                        y=df_factory['Production'],
                        name=factory
                    ),
                    row=1, col=1
                )
            
            # Add shipping plan
            shipping_data = self.solution['shipping']
            df_ship = pd.DataFrame([
                {
                    'From': f'Factory {i}',
                    'To': f'Customer {j}',
                    'Period': t,
                    'Units': units
                }
                for (i, j, t), units in shipping_data.items()
            ])
            
            for factory in df_ship['From'].unique():
                df_factory = df_ship[df_ship['From'] == factory]
                fig.add_trace(
                    go.Bar(
                        x=df_factory['Period'],
                        y=df_factory['Units'],
                        name=factory
                    ),
                    row=1, col=2
                )
            
            # Add inventory levels
            inventory_data = self.solution['inventory']
            df_inv = pd.DataFrame([
                {'Factory': f'Factory {i}', 'Period': t, 'Inventory': inv}
                for (i, t), inv in inventory_data.items()
            ])
            
            for factory in df_inv['Factory'].unique():
                df_factory = df_inv[df_inv['Factory'] == factory]
                fig.add_trace(
                    go.Scatter(
                        x=df_factory['Period'],
                        y=df_factory['Inventory'],
                        name=factory,
                        mode='lines+markers'
                    ),
                    row=2, col=1
                )
            
            # Add cost breakdown
            production_cost = sum(
                self.solution['production'].get((i, t), 0)
                for i in range(3) for t in range(3)
            )
            
            shipping_cost = sum(
                self.solution['shipping'].get((i, j, t), 0)
                for i in range(3) for j in range(3) for t in range(3)
            )
            
            storage_cost = sum(
                self.solution['inventory'].get((i, t), 0)
                for i in range(3) for t in range(3)
            )
            
            fig.add_trace(
                go.Pie(
                    labels=['Production', 'Shipping', 'Storage'],
                    values=[production_cost, shipping_cost, storage_cost],
                    name='Cost Breakdown'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                title_text='Supply Chain Optimization Dashboard',
                showlegend=True
            )
            
            # Save dashboard
            fig.write_html(os.path.join(self.output_dir, 'supply_chain_dashboard.html'))
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise

    def plot_optimization_convergence(self,
                                    objective_values: List[float],
                                    title: str = "Optimization Convergence") -> None:
        """
        Plot the convergence of the optimization algorithm.
        
        Args:
            objective_values: List of objective function values
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.plot(objective_values)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.grid(True)
        plt.tight_layout()
        plt.show() 