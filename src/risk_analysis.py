import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from supply_chain_optimization import SupplyChainOptimizer
import logging
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class RiskType(Enum):
    """Types of supply chain risks."""
    SUPPLY = "supply"
    DEMAND = "demand"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    EXTERNAL = "external"

@dataclass
class Risk:
    """Class to represent a supply chain risk."""
    name: str
    type: RiskType
    probability: float
    impact: float
    description: str
    mitigation_strategies: List[str]

class RiskAnalyzer:
    def __init__(self, optimizer: SupplyChainOptimizer):
        """
        Initialize the Risk Analyzer.
        
        Args:
            optimizer: SupplyChainOptimizer instance
        """
        self.optimizer = optimizer
        self.risks = []
        self.risk_matrix = None
        
    def add_risk(self, risk: Risk) -> None:
        """
        Add a risk to analyze.
        
        Args:
            risk: Risk object to add
        """
        self.risks.append(risk)
        
    def create_default_risks(self) -> None:
        """Create a set of default supply chain risks."""
        # Supply risks
        self.add_risk(Risk(
            name="Supplier Disruption",
            type=RiskType.SUPPLY,
            probability=0.3,
            impact=0.7,
            description="Disruption in supplier operations affecting raw material availability",
            mitigation_strategies=["Diversify supplier base", "Maintain safety stock", "Develop alternative suppliers"]
        ))
        
        self.add_risk(Risk(
            name="Quality Issues",
            type=RiskType.SUPPLY,
            probability=0.2,
            impact=0.6,
            description="Quality problems with supplied materials",
            mitigation_strategies=["Implement quality control", "Regular supplier audits", "Quality agreements"]
        ))
        
        # Demand risks
        self.add_risk(Risk(
            name="Demand Volatility",
            type=RiskType.DEMAND,
            probability=0.4,
            impact=0.5,
            description="Unexpected changes in customer demand",
            mitigation_strategies=["Improve forecasting", "Flexible production", "Inventory buffers"]
        ))
        
        self.add_risk(Risk(
            name="Market Competition",
            type=RiskType.DEMAND,
            probability=0.3,
            impact=0.6,
            description="Increased competition affecting market share",
            mitigation_strategies=["Product differentiation", "Customer loyalty programs", "Market analysis"]
        ))
        
        # Operational risks
        self.add_risk(Risk(
            name="Equipment Failure",
            type=RiskType.OPERATIONAL,
            probability=0.2,
            impact=0.8,
            description="Breakdown of production equipment",
            mitigation_strategies=["Preventive maintenance", "Backup equipment", "Staff training"]
        ))
        
        self.add_risk(Risk(
            name="Labor Shortage",
            type=RiskType.OPERATIONAL,
            probability=0.3,
            impact=0.6,
            description="Shortage of skilled labor",
            mitigation_strategies=["Cross-training", "Automation", "Employee retention programs"]
        ))
        
        # Financial risks
        self.add_risk(Risk(
            name="Cost Inflation",
            type=RiskType.FINANCIAL,
            probability=0.4,
            impact=0.5,
            description="Rising costs of materials and operations",
            mitigation_strategies=["Cost optimization", "Long-term contracts", "Efficiency improvements"]
        ))
        
        self.add_risk(Risk(
            name="Currency Fluctuation",
            type=RiskType.FINANCIAL,
            probability=0.3,
            impact=0.4,
            description="Exchange rate fluctuations affecting costs",
            mitigation_strategies=["Hedging strategies", "Local sourcing", "Price adjustments"]
        ))
        
        # External risks
        self.add_risk(Risk(
            name="Natural Disasters",
            type=RiskType.EXTERNAL,
            probability=0.1,
            impact=0.9,
            description="Natural disasters affecting operations",
            mitigation_strategies=["Geographic diversification", "Disaster recovery plans", "Insurance coverage"]
        ))
        
        self.add_risk(Risk(
            name="Regulatory Changes",
            type=RiskType.EXTERNAL,
            probability=0.2,
            impact=0.7,
            description="Changes in regulations affecting operations",
            mitigation_strategies=["Regulatory monitoring", "Compliance programs", "Government relations"]
        ))
    
    def calculate_risk_scores(self) -> pd.DataFrame:
        """
        Calculate risk scores for all risks.
        
        Returns:
            DataFrame with risk scores
        """
        risk_data = []
        
        for risk in self.risks:
            risk_score = risk.probability * risk.impact
            risk_data.append({
                'risk_name': risk.name,
                'risk_type': risk.type.value,
                'probability': risk.probability,
                'impact': risk.impact,
                'risk_score': risk_score,
                'description': risk.description,
                'mitigation_strategies': ', '.join(risk.mitigation_strategies)
            })
        
        return pd.DataFrame(risk_data)
    
    def create_risk_matrix(self) -> None:
        """Create a risk matrix for visualization."""
        risk_scores = self.calculate_risk_scores()
        
        # Create risk matrix
        plt.figure(figsize=(10, 8))
        plt.scatter(risk_scores['probability'], risk_scores['impact'],
                   s=risk_scores['risk_score']*1000, alpha=0.6)
        
        # Add labels
        for i, row in risk_scores.iterrows():
            plt.annotate(row['risk_name'],
                        (row['probability'], row['impact']),
                        xytext=(5, 5), textcoords='offset points')
        
        # Add grid and labels
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Probability')
        plt.ylabel('Impact')
        plt.title('Supply Chain Risk Matrix')
        
        # Add risk zones
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.3)
        
        # Add zone labels
        plt.text(0.25, 0.25, 'Low Risk', ha='center', va='center')
        plt.text(0.75, 0.25, 'Medium Risk', ha='center', va='center')
        plt.text(0.25, 0.75, 'Medium Risk', ha='center', va='center')
        plt.text(0.75, 0.75, 'High Risk', ha='center', va='center')
        
        self.risk_matrix = plt.gcf()
    
    def get_risk_summary(self) -> pd.DataFrame:
        """
        Generate a summary of risk analysis.
        
        Returns:
            DataFrame with risk summary
        """
        risk_scores = self.calculate_risk_scores()
        
        summary = risk_scores.groupby('risk_type').agg({
            'risk_score': ['mean', 'std', 'min', 'max'],
            'probability': ['mean', 'std'],
            'impact': ['mean', 'std']
        }).round(2)
        
        return summary
    
    def get_high_risk_areas(self) -> pd.DataFrame:
        """
        Identify high-risk areas based on risk scores.
        
        Returns:
            DataFrame containing high-risk areas
        """
        high_risk_threshold = 0.15  # Lowered from 0.2 to 0.15
        high_risks = self.risk_df[self.risk_df['risk_score'] >= high_risk_threshold]
        return high_risks[['risk_name', 'risk_type', 'risk_score', 'description']]
        
    def get_mitigation_recommendations(self) -> Dict[str, List[str]]:
        """
        Generate mitigation recommendations for each risk type.
        
        Returns:
            Dictionary mapping risk types to lists of mitigation strategies
        """
        recommendations = {
            'supply': [
                "Diversify supplier base",
                "Implement supplier performance monitoring",
                "Develop backup suppliers",
                "Maintain safety stock levels",
                "Establish long-term supplier relationships"
            ],
            'demand': [
                "Improve demand forecasting accuracy",
                "Implement flexible production systems",
                "Develop responsive inventory management",
                "Create demand-driven supply chain",
                "Establish customer feedback mechanisms"
            ],
            'operational': [
                "Implement preventive maintenance programs",
                "Develop contingency plans",
                "Train staff on risk management",
                "Optimize production scheduling",
                "Implement quality control measures"
            ],
            'financial': [
                "Implement cost control measures",
                "Develop financial risk management strategies",
                "Optimize working capital",
                "Monitor market conditions",
                "Establish cost reduction initiatives"
            ],
            'external': [
                "Monitor regulatory changes",
                "Develop disaster recovery plans",
                "Implement environmental risk management",
                "Establish political risk monitoring",
                "Create crisis management protocols"
            ]
        }
        
        # Add specific recommendations based on high-risk areas
        high_risks = self.get_high_risk_areas()
        for _, risk in high_risks.iterrows():
            risk_type = risk['risk_type']
            if risk_type in recommendations:
                # Add specific mitigation strategy based on risk description
                specific_strategy = f"Address {risk['risk_name'].lower()} through targeted mitigation"
                if specific_strategy not in recommendations[risk_type]:
                    recommendations[risk_type].append(specific_strategy)
        
        return recommendations 