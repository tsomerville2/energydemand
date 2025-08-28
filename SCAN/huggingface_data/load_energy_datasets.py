"""
HuggingFace Energy Datasets Demo
Load and analyze real energy consumption datasets from HuggingFace Hub
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("HUGGINGFACE ENERGY DATASETS")
print("=" * 60)

# Load electricity load diagrams dataset (Portuguese clients)
print("\n1. Loading Electricity Load Diagrams Dataset...")
print("   This contains hourly kW consumption for 370 Portuguese clients")

try:
    # Load a subset of the dataset to avoid memory issues
    dataset = load_dataset("electricity_load_diagrams", split="train[:1000]")
    print(f"   Successfully loaded {len(dataset)} samples")
    
    # Convert to pandas dataframe for easier manipulation
    df = pd.DataFrame(dataset)
    
    # Dataset info
    print("\nDataset Structure:")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Shape: {df.shape}")
    print(f"  First timestamp: {df['start'].iloc[0]}")
    print(f"  Last timestamp: {df['start'].iloc[-1]}")
    
    # Extract MT_001 client data (first client)
    client_cols = [col for col in df.columns if col.startswith('MT_')]
    print(f"\n  Number of clients: {len(client_cols)}")
    print(f"  Sample clients: {client_cols[:5]}")
    
    # Analyze first client
    client_1 = df[['start', client_cols[0]]].copy()
    client_1.columns = ['timestamp', 'consumption_kw']
    client_1['timestamp'] = pd.to_datetime(client_1['timestamp'])
    client_1 = client_1.set_index('timestamp').sort_index()
    
    print(f"\nClient {client_cols[0]} Statistics:")
    print(f"  Mean consumption: {client_1['consumption_kw'].mean():.2f} kW")
    print(f"  Max consumption: {client_1['consumption_kw'].max():.2f} kW")
    print(f"  Min consumption: {client_1['consumption_kw'].min():.2f} kW")
    print(f"  Std deviation: {client_1['consumption_kw'].std():.2f} kW")
    
    # Basic time series analysis
    client_1['hour'] = client_1.index.hour
    client_1['day_of_week'] = client_1.index.dayofweek
    hourly_avg = client_1.groupby('hour')['consumption_kw'].mean()
    daily_avg = client_1.groupby('day_of_week')['consumption_kw'].mean()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Raw time series
    ax = axes[0, 0]
    ax.plot(client_1.index[:168], client_1['consumption_kw'][:168], linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Consumption (kW)')
    ax.set_title(f'One Week of Hourly Consumption - {client_cols[0]}')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Daily profile
    ax = axes[0, 1]
    ax.plot(hourly_avg.index, hourly_avg.values, linewidth=2, marker='o')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Consumption (kW)')
    ax.set_title('Average Daily Load Profile')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 3))
    
    # Plot 3: Weekly pattern
    ax = axes[1, 0]
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.bar(range(7), daily_avg.values, color='steelblue', alpha=0.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(days)
    ax.set_ylabel('Average Consumption (kW)')
    ax.set_title('Average Consumption by Day of Week')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Load duration curve
    ax = axes[1, 1]
    sorted_load = np.sort(client_1['consumption_kw'].values)[::-1]
    duration_pct = np.linspace(0, 100, len(sorted_load))
    ax.plot(duration_pct, sorted_load, linewidth=2)
    ax.set_xlabel('Duration (%)')
    ax.set_ylabel('Load (kW)')
    ax.set_title('Load Duration Curve')
    ax.grid(True, alpha=0.3)
    ax.fill_between(duration_pct, 0, sorted_load, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('SCAN/huggingface_data/electricity_load_analysis.png', dpi=150, bbox_inches='tight')
    
    # Save sample data
    client_1.to_csv('SCAN/huggingface_data/sample_client_data.csv')
    
    print("\n✅ Electricity Load Diagrams dataset successfully loaded and analyzed")
    
except Exception as e:
    print(f"\n❌ Error loading electricity_load_diagrams: {e}")

# Try loading ETT dataset (Electricity Transformer Temperature)
print("\n2. Loading ETT (Electricity Transformer Temperature) Dataset...")

try:
    ett_dataset = load_dataset("ett", "h1", split="train[:500]")
    print(f"   Successfully loaded {len(ett_dataset)} samples")
    
    ett_df = pd.DataFrame(ett_dataset)
    print("\nETT Dataset Structure:")
    print(f"  Columns: {ett_df.columns.tolist()}")
    print(f"  Shape: {ett_df.shape}")
    
    # Basic statistics
    print("\nETT Dataset Statistics:")
    print(f"  Mean OT (Oil Temperature): {ett_df['OT'].mean():.2f}")
    print(f"  Mean HUFL (High UseFul Load): {ett_df['HUFL'].mean():.2f}")
    print(f"  Mean HULL (High UseLess Load): {ett_df['HULL'].mean():.2f}")
    
    # Save sample
    ett_df.head(100).to_csv('SCAN/huggingface_data/ett_sample.csv', index=False)
    print("\n✅ ETT dataset successfully loaded")
    
except Exception as e:
    print(f"\n❌ Error loading ETT dataset: {e}")

# Simple forecasting baseline
print("\n3. Creating Simple Forecasting Baseline...")

try:
    if 'client_1' in locals():
        # Use last 24 hours to predict next 24 hours
        train_size = len(client_1) - 24
        train_data = client_1['consumption_kw'][:train_size]
        test_data = client_1['consumption_kw'][train_size:]
        
        # Naive forecast (repeat last day)
        naive_forecast = train_data[-24:].values
        
        # Moving average forecast
        ma_forecast = train_data.rolling(window=24).mean().iloc[-1]
        ma_forecast = np.repeat(ma_forecast, 24)
        
        # Calculate errors
        if len(test_data) >= 24:
            test_24h = test_data[:24].values
            naive_mape = np.mean(np.abs((test_24h - naive_forecast) / test_24h)) * 100
            ma_mape = np.mean(np.abs((test_24h - ma_forecast) / test_24h)) * 100
            
            print(f"Forecasting Results (24-hour ahead):")
            print(f"  Naive (repeat last day) MAPE: {naive_mape:.1f}%")
            print(f"  Moving Average MAPE: {ma_mape:.1f}%")
            
            # Plot forecast comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            hours = range(24)
            ax.plot(hours, test_24h, label='Actual', linewidth=2, marker='o', markersize=4)
            ax.plot(hours, naive_forecast, label='Naive Forecast', linewidth=2, linestyle='--', alpha=0.7)
            ax.plot(hours, ma_forecast, label='MA Forecast', linewidth=2, linestyle=':', alpha=0.7)
            ax.set_xlabel('Hour')
            ax.set_ylabel('Consumption (kW)')
            ax.set_title('24-Hour Ahead Forecasting Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig('SCAN/huggingface_data/forecast_comparison.png', dpi=150, bbox_inches='tight')
            
            print("\n✅ Forecasting baseline created")
            
except Exception as e:
    print(f"\n❌ Error creating forecast: {e}")

print("\n" + "=" * 60)
print("HuggingFace Energy Datasets Analysis Complete!")
print("Results saved to:")
print("  - SCAN/huggingface_data/electricity_load_analysis.png")
print("  - SCAN/huggingface_data/sample_client_data.csv")
print("  - SCAN/huggingface_data/ett_sample.csv")
print("  - SCAN/huggingface_data/forecast_comparison.png")
print("=" * 60)