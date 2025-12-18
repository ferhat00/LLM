import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WineRegionalAnalysis:
    def __init__(self):
        """Initialize the wine regional analysis with sample data"""
        self.bordeaux_data = self._generate_bordeaux_data()
        self.burgundy_data = self._generate_burgundy_data()
        self.combined_data = self._combine_regional_data()
    
    def _generate_bordeaux_data(self):
        """Generate realistic Bordeaux market data for analysis"""
        dates = pd.date_range(start='2024-01-01', end='2025-05-31', freq='M')
        
        # Bordeaux price trends (showing decline)
        base_price = 1000
        price_trend = np.linspace(0, -15, len(dates))  # 15% decline over period
        price_volatility = np.random.normal(0, 2, len(dates))
        prices = base_price * (1 + (price_trend + price_volatility) / 100)
        
        # Production volumes (higher volume, consistent)
        base_volume = 850000  # hectoliters
        volume_variation = np.random.normal(0, 5, len(dates))
        volumes = base_volume * (1 + volume_variation / 100)
        
        # Export data
        base_exports = 45000  # thousands of bottles
        export_decline = np.linspace(0, -12, len(dates))  # Export decline
        export_volatility = np.random.normal(0, 3, len(dates))
        exports = base_exports * (1 + (export_decline + export_volatility) / 100)
        
        return pd.DataFrame({
            'date': dates,
            'region': 'Bordeaux',
            'avg_price_eur': prices,
            'production_hl': volumes,
            'exports_k_bottles': exports,
            'inventory_months': np.random.uniform(8, 14, len(dates)),
            'investment_score': np.random.uniform(6.5, 8.2, len(dates))
        })
    
    def _generate_burgundy_data(self):
        """Generate realistic Burgundy market data for analysis"""
        dates = pd.date_range(start='2024-01-01', end='2025-05-31', freq='M')
        
        # Burgundy price trends (showing resilience/growth)
        base_price = 2500
        price_trend = np.linspace(0, 8, len(dates))  # 8% growth over period
        price_volatility = np.random.normal(0, 4, len(dates))
        prices = base_price * (1 + (price_trend + price_volatility) / 100)
        
        # Production volumes (lower volume, more volatile due to climate)
        base_volume = 180000  # hectoliters (much lower than Bordeaux)
        # Simulate 2024 production crisis
        volume_shock = np.where(dates >= '2024-08-01', -35, 0)  # 35% decline post-harvest
        volume_volatility = np.random.normal(0, 12, len(dates))
        volumes = base_volume * (1 + (volume_shock + volume_volatility) / 100)
        
        # Export data (more stable, premium market)
        base_exports = 12000  # thousands of bottles (much lower volume)
        export_trend = np.linspace(0, 5, len(dates))  # Slight growth
        export_volatility = np.random.normal(0, 2, len(dates))
        exports = base_exports * (1 + (export_trend + export_volatility) / 100)
        
        return pd.DataFrame({
            'date': dates,
            'region': 'Burgundy',
            'avg_price_eur': prices,
            'production_hl': volumes,
            'exports_k_bottles': exports,
            'inventory_months': np.random.uniform(3, 8, len(dates)),  # Lower inventory
            'investment_score': np.random.uniform(8.5, 9.5, len(dates))  # Higher scores
        })
    
    def _combine_regional_data(self):
        """Combine Bordeaux and Burgundy data for comparative analysis"""
        combined = pd.concat([self.bordeaux_data, self.burgundy_data], ignore_index=True)
        
        # Calculate additional metrics
        combined['price_per_bottle'] = combined['avg_price_eur']
        combined['volume_per_export'] = combined['production_hl'] / combined['exports_k_bottles']
        combined['scarcity_index'] = 1 / (combined['inventory_months'] / 12)
        
        # Calculate rolling metrics
        for region in ['Bordeaux', 'Burgundy']:
            mask = combined['region'] == region
            combined.loc[mask, 'price_volatility_3m'] = (
                combined.loc[mask, 'avg_price_eur']
                .rolling(window=3, min_periods=1)
                .std()
            )
            combined.loc[mask, 'price_change_12m'] = (
                combined.loc[mask, 'avg_price_eur']
                .pct_change(periods=12) * 100
            )
        
        return combined
    
    def plot_price_comparison(self):
        """Create comprehensive price comparison visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price trends over time
        for region in ['Bordeaux', 'Burgundy']:
            data = self.combined_data[self.combined_data['region'] == region]
            ax1.plot(data['date'], data['avg_price_eur'], 
                    marker='o', linewidth=2.5, label=region, markersize=4)
        
        ax1.set_title('Average Wine Prices: Bordeaux vs Burgundy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (EUR)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Price volatility comparison
        bordeaux_vol = self.bordeaux_data['avg_price_eur'].std()
        burgundy_vol = self.burgundy_data['avg_price_eur'].std()
        
        ax2.bar(['Bordeaux', 'Burgundy'], [bordeaux_vol, burgundy_vol], 
                color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax2.set_title('Price Volatility Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Standard Deviation (EUR)', fontsize=12)
        
        # Production volumes
        sns.boxplot(data=self.combined_data, x='region', y='production_hl', ax=ax3)
        ax3.set_title('Production Volume Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Production (Hectoliters)', fontsize=12)
        ax3.set_xlabel('Region', fontsize=12)
        
        # Investment scores over time
        for region in ['Bordeaux', 'Burgundy']:
            data = self.combined_data[self.combined_data['region'] == region]
            ax4.plot(data['date'], data['investment_score'], 
                    marker='s', linewidth=2, label=region, markersize=3)
        
        ax4.set_title('Investment Attractiveness Score', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Investment Score (1-10)', fontsize=12)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_market_metrics(self):
        """Calculate and display key market metrics"""
        metrics = {}
        
        for region in ['Bordeaux', 'Burgundy']:
            data = self.combined_data[self.combined_data['region'] == region]
            
            metrics[region] = {
                'avg_price': data['avg_price_eur'].mean(),
                'price_volatility': data['avg_price_eur'].std(),
                'price_growth': ((data['avg_price_eur'].iloc[-1] / data['avg_price_eur'].iloc[0]) - 1) * 100,
                'avg_production': data['production_hl'].mean(),
                'production_volatility': data['production_hl'].std() / data['production_hl'].mean() * 100,
                'avg_exports': data['exports_k_bottles'].mean(),
                'avg_inventory': data['inventory_months'].mean(),
                'scarcity_index': data['scarcity_index'].mean(),
                'investment_score': data['investment_score'].mean()
            }
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(metrics).T
        comparison_df = comparison_df.round(2)
        
        print("=== REGIONAL WINE MARKET ANALYSIS ===\n")
        print("Key Performance Metrics Comparison:")
        print("=" * 50)
        print(comparison_df.to_string())
        
        # Calculate additional insights
        print(f"\n=== MARKET INSIGHTS ===")
        
        # Price premium analysis
        burgundy_premium = (metrics['Burgundy']['avg_price'] / metrics['Bordeaux']['avg_price'] - 1) * 100
        print(f"Burgundy Price Premium: {burgundy_premium:.1f}%")
        
        # Production comparison
        production_ratio = metrics['Bordeaux']['avg_production'] / metrics['Burgundy']['avg_production']
        print(f"Bordeaux produces {production_ratio:.1f}x more volume than Burgundy")
        
        # Risk-return analysis
        bordeaux_risk_return = metrics['Bordeaux']['price_growth'] / metrics['Bordeaux']['price_volatility']
        burgundy_risk_return = metrics['Burgundy']['price_growth'] / metrics['Burgundy']['price_volatility']
        
        print(f"\nRisk-Adjusted Returns (Growth/Volatility):")
        print(f"Bordeaux: {bordeaux_risk_return:.3f}")
        print(f"Burgundy: {burgundy_risk_return:.3f}")
        
        if burgundy_risk_return > bordeaux_risk_return:
            print("→ Burgundy shows superior risk-adjusted performance")
        else:
            print("→ Bordeaux shows superior risk-adjusted performance")
        
        return comparison_df
    
    def plot_strategic_positioning(self):
        """Create strategic positioning matrix"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Volume vs Price positioning
        for region in ['Bordeaux', 'Burgundy']:
            data = self.combined_data[self.combined_data['region'] == region]
            avg_price = data['avg_price_eur'].mean()
            avg_volume = data['production_hl'].mean()
            
            ax1.scatter(avg_volume, avg_price, s=200, alpha=0.7, label=region)
            ax1.annotate(region, (avg_volume, avg_price), 
                        xytext=(10, 10), textcoords='offset points', fontsize=12)
        
        ax1.set_xlabel('Average Production Volume (Hectoliters)', fontsize=12)
        ax1.set_ylabel('Average Price (EUR)', fontsize=12)
        ax1.set_title('Strategic Positioning: Volume vs Price', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Scarcity vs Investment Score
        for region in ['Bordeaux', 'Burgundy']:
            data = self.combined_data[self.combined_data['region'] == region]
            avg_scarcity = data['scarcity_index'].mean()
            avg_investment = data['investment_score'].mean()
            
            ax2.scatter(avg_scarcity, avg_investment, s=200, alpha=0.7, label=region)
            ax2.annotate(region, (avg_scarcity, avg_investment), 
                        xytext=(10, 10), textcoords='offset points', fontsize=12)
        
        ax2.set_xlabel('Scarcity Index', fontsize=12)
        ax2.set_ylabel('Investment Score', fontsize=12)
        ax2.set_title('Investment Analysis: Scarcity vs Attractiveness', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_analysis_data(self, filename='wine_regional_analysis.csv'):
        """Export analysis data to CSV"""
        self.combined_data.to_csv(filename, index=False)
        print(f"Analysis data exported to {filename}")
        
        # Export summary statistics
        summary_stats = self.combined_data.groupby('region').agg({
            'avg_price_eur': ['mean', 'std', 'min', 'max'],
            'production_hl': ['mean', 'std', 'min', 'max'],
            'exports_k_bottles': ['mean', 'std'],
            'investment_score': ['mean', 'std']
        }).round(2)
        
        summary_filename = 'wine_summary_statistics.csv'
        summary_stats.to_csv(summary_filename)
        print(f"Summary statistics exported to {summary_filename}")
    
    def run_complete_analysis(self):
        """Run the complete regional analysis"""
        print("Starting Bordeaux vs Burgundy Regional Analysis...")
        print("=" * 55)
        
        # Market metrics analysis
        comparison_df = self.analyze_market_metrics()
        
        # Visualizations
        print(f"\nGenerating visualizations...")
        self.plot_price_comparison()
        self.plot_strategic_positioning()
        
        # Export data
        self.export_analysis_data()
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print("Key findings:")
        print("• Price trends, production volumes, and investment metrics analyzed")
        print("• Strategic positioning matrices created")
        print("• Data exported for further analysis")
        
        return comparison_df

# Example usage and execution
if __name__ == "__main__":
    # Initialize and run analysis
    analyzer = WineRegionalAnalysis()
    results = analyzer.run_complete_analysis()
    
    # Additional custom analysis example
    print(f"\n=== CUSTOM ANALYSIS EXAMPLES ===")
    
    # Correlation analysis
    bordeaux_data = analyzer.bordeaux_data
    burgundy_data = analyzer.burgundy_data
    
    # Price-production correlation
    bordeaux_corr = bordeaux_data['avg_price_eur'].corr(bordeaux_data['production_hl'])
    burgundy_corr = burgundy_data['avg_price_eur'].corr(burgundy_data['production_hl'])
    
    print(f"Price-Production Correlation:")
    print(f"Bordeaux: {bordeaux_corr:.3f}")
    print(f"Burgundy: {burgundy_corr:.3f}")
    
    # Market efficiency analysis
    print(f"\nMarket Efficiency Indicators:")
    print(f"Bordeaux price volatility: {bordeaux_data['avg_price_eur'].std():.2f}")
    print(f"Burgundy price volatility: {burgundy_data['avg_price_eur'].std():.2f}")
    
    if burgundy_data['avg_price_eur'].std() > bordeaux_data['avg_price_eur'].std():
        print("→ Burgundy shows higher price volatility (potentially less efficient market)")
    else:
        print("→ Bordeaux shows higher price volatility (potentially less efficient market)")