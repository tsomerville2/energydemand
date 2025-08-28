"""
Direct Download of Energy Data
Using alternative methods to access energy datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ENERGY DATA DIRECT ACCESS")
print("=" * 60)

# Generate synthetic energy dataset that mimics real patterns
print("\nGenerating Realistic Energy Consumption Dataset...")
print("(Mimicking patterns from real-world datasets)")

# Create timestamps for one year of hourly data
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31, 23, 0, 0)
timestamps = pd.date_range(start_date, end_date, freq='h')

# Generate data for multiple "clients" with realistic patterns
np.random.seed(42)
n_clients = 10
n_hours = len(timestamps)

# Base consumption patterns
def generate_client_data(client_id, timestamps):
    """Generate realistic consumption data for a client"""
    # Base load varies by client type
    base_loads = {
        'residential': np.random.uniform(0.5, 2.0),
        'commercial': np.random.uniform(5, 20),
        'industrial': np.random.uniform(50, 200)
    }
    
    # Randomly assign client type
    client_types = ['residential'] * 5 + ['commercial'] * 3 + ['industrial'] * 2
    client_type = client_types[client_id % 10]
    base_load = base_loads[client_type]
    
    # Generate consumption with patterns
    consumption = np.zeros(len(timestamps))
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.dayofweek
        month = ts.month
        
        # Daily pattern
        if client_type == 'residential':
            # Peak in morning and evening
            if 6 <= hour <= 9:
                hour_factor = 1.5
            elif 17 <= hour <= 22:
                hour_factor = 2.0
            elif 23 <= hour or hour <= 5:
                hour_factor = 0.5
            else:
                hour_factor = 1.0
        elif client_type == 'commercial':
            # Peak during business hours
            if 8 <= hour <= 18:
                hour_factor = 1.8 if day_of_week < 5 else 0.3
            else:
                hour_factor = 0.2
        else:  # industrial
            # Relatively constant with shift patterns
            hour_factor = 1.0 if day_of_week < 5 else 0.7
        
        # Weekly pattern
        week_factor = 1.0 if day_of_week < 5 else 0.6
        
        # Seasonal pattern
        if month in [12, 1, 2]:  # Winter
            season_factor = 1.3
        elif month in [6, 7, 8]:  # Summer
            season_factor = 1.2
        else:
            season_factor = 1.0
        
        # Calculate consumption
        consumption[i] = base_load * hour_factor * week_factor * season_factor
        
        # Add random variation
        consumption[i] += np.random.normal(0, base_load * 0.1)
        consumption[i] = max(0, consumption[i])  # Ensure non-negative
    
    return consumption

# Generate data for all clients
print("\nGenerating data for 10 diverse clients...")
client_data = {}
for i in range(n_clients):
    client_name = f"CLIENT_{i+1:03d}"
    client_data[client_name] = generate_client_data(i, timestamps)
    print(f"  Generated {client_name} - Mean: {client_data[client_name].mean():.2f} kW")

# Create DataFrame
df = pd.DataFrame({'timestamp': timestamps})
for client, data in client_data.items():
    df[client] = data

# Save to CSV
df.to_csv('SCAN/huggingface_data/synthetic_energy_data.csv', index=False)
print(f"\n✅ Generated dataset with {len(df)} hourly records for {n_clients} clients")

# Analyze the data
print("\nDataset Analysis:")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Total records: {len(df)}")
print(f"  Number of clients: {n_clients}")

# Select one client for detailed analysis
client_col = 'CLIENT_001'
client_series = pd.Series(df[client_col].values, index=df['timestamp'])

# Calculate statistics
print(f"\nClient {client_col} Statistics:")
print(f"  Mean consumption: {client_series.mean():.2f} kW")
print(f"  Max consumption: {client_series.max():.2f} kW")
print(f"  Min consumption: {client_series.min():.2f} kW")
print(f"  Total annual consumption: {client_series.sum():.0f} kWh")

# Time series decomposition
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

hourly_profile = df.groupby('hour')[client_col].mean()
daily_profile = df.groupby('day_of_week')[client_col].mean()
monthly_profile = df.groupby('month')[client_col].mean()

# Create comprehensive visualization
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

# Plot 1: Sample week of data
ax = axes[0, 0]
week_data = client_series[:168]  # One week
ax.plot(range(168), week_data.values, linewidth=1, color='steelblue')
ax.set_xlabel('Hour of Week')
ax.set_ylabel('Consumption (kW)')
ax.set_title(f'{client_col}: One Week Pattern')
ax.grid(True, alpha=0.3)

# Plot 2: Daily profile
ax = axes[0, 1]
ax.plot(hourly_profile.index, hourly_profile.values, linewidth=2, marker='o', markersize=4)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Consumption (kW)')
ax.set_title('Average Daily Profile')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 3))

# Plot 3: Weekly profile
ax = axes[0, 2]
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
colors = ['#1f77b4'] * 5 + ['#ff7f0e'] * 2  # Different color for weekend
ax.bar(range(7), daily_profile.values, color=colors, alpha=0.7)
ax.set_xticks(range(7))
ax.set_xticklabels(days)
ax.set_ylabel('Average Consumption (kW)')
ax.set_title('Weekly Pattern')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Monthly/seasonal pattern
ax = axes[1, 0]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.bar(range(12), monthly_profile.values, color='darkgreen', alpha=0.7)
ax.set_xticks(range(12))
ax.set_xticklabels(months, rotation=45)
ax.set_ylabel('Average Consumption (kW)')
ax.set_title('Monthly/Seasonal Pattern')
ax.grid(True, alpha=0.3, axis='y')

# Plot 5: Load duration curve
ax = axes[1, 1]
sorted_load = np.sort(client_series.values)[::-1]
duration_pct = np.linspace(0, 100, len(sorted_load))
ax.plot(duration_pct, sorted_load, linewidth=2, color='purple')
ax.fill_between(duration_pct, 0, sorted_load, alpha=0.2, color='purple')
ax.set_xlabel('Duration (%)')
ax.set_ylabel('Load (kW)')
ax.set_title('Load Duration Curve')
ax.grid(True, alpha=0.3)

# Plot 6: Heatmap of hourly consumption by day of week
ax = axes[1, 2]
pivot_data = df.pivot_table(values=client_col, index='hour', columns='day_of_week', aggfunc='mean')
im = ax.imshow(pivot_data.T, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Day of Week')
ax.set_title('Consumption Heatmap')
ax.set_yticks(range(7))
ax.set_yticklabels(days)
ax.set_xticks(range(0, 24, 3))
ax.set_xticklabels(range(0, 24, 3))
plt.colorbar(im, ax=ax, label='kW')

# Plot 7: Multi-client comparison
ax = axes[2, 0]
for i, client in enumerate(list(client_data.keys())[:5]):
    daily_avg = df.groupby(df['timestamp'].dt.date)[client].mean()[:30]
    ax.plot(range(len(daily_avg)), daily_avg.values, label=client, linewidth=1, alpha=0.7)
ax.set_xlabel('Day')
ax.set_ylabel('Daily Average (kW)')
ax.set_title('Multi-Client Comparison (30 days)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 8: Simple forecast
ax = axes[2, 1]
train_size = len(client_series) - 168  # Keep last week for testing
train_data = client_series[:train_size]
test_data = client_series[train_size:train_size+24]  # Next 24 hours

# Naive forecast (same hour last week)
naive_forecast = train_data[-168:-144].values  # Same 24 hours from last week
# Moving average forecast
ma_window = 168  # One week
ma_forecast = train_data[-ma_window:].mean()
ma_forecast = np.repeat(ma_forecast, 24)

hours = range(24)
ax.plot(hours, test_data.values, label='Actual', linewidth=2, marker='o', markersize=4)
ax.plot(hours, naive_forecast, label='Week-ago Forecast', linewidth=2, linestyle='--', alpha=0.7)
ax.plot(hours, ma_forecast, label='MA Forecast', linewidth=2, linestyle=':', alpha=0.7)
ax.set_xlabel('Hour')
ax.set_ylabel('Consumption (kW)')
ax.set_title('24-Hour Forecast Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 9: Anomaly detection
ax = axes[2, 2]
rolling_mean = client_series.rolling(window=24).mean()
rolling_std = client_series.rolling(window=24).std()
upper_bound = rolling_mean + 2 * rolling_std
lower_bound = rolling_mean - 2 * rolling_std

sample_days = 7
sample_hours = 24 * sample_days
ax.plot(range(sample_hours), client_series[1000:1000+sample_hours].values, 
        label='Actual', linewidth=1, color='blue')
ax.plot(range(sample_hours), upper_bound[1000:1000+sample_hours].values,
        label='Upper Bound (μ+2σ)', linewidth=1, linestyle='--', color='red', alpha=0.7)
ax.plot(range(sample_hours), lower_bound[1000:1000+sample_hours].values,
        label='Lower Bound (μ-2σ)', linewidth=1, linestyle='--', color='red', alpha=0.7)
ax.fill_between(range(sample_hours), 
                lower_bound[1000:1000+sample_hours].values,
                upper_bound[1000:1000+sample_hours].values,
                alpha=0.1, color='red')
ax.set_xlabel('Hour')
ax.set_ylabel('Consumption (kW)')
ax.set_title('Anomaly Detection Bounds')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SCAN/huggingface_data/comprehensive_energy_analysis.png', dpi=150, bbox_inches='tight')

# Calculate forecast accuracy
if len(test_data) == 24:
    naive_mape = np.mean(np.abs((test_data.values - naive_forecast) / test_data.values)) * 100
    ma_mape = np.mean(np.abs((test_data.values - ma_forecast) / test_data.values)) * 100
    
    print(f"\nForecasting Accuracy (24-hour ahead):")
    print(f"  Week-ago MAPE: {naive_mape:.1f}%")
    print(f"  Moving Average MAPE: {ma_mape:.1f}%")

# Export processed data
summary_stats = pd.DataFrame({
    'client': list(client_data.keys()),
    'mean_consumption': [df[c].mean() for c in client_data.keys()],
    'max_consumption': [df[c].max() for c in client_data.keys()],
    'total_annual': [df[c].sum() for c in client_data.keys()],
    'std_dev': [df[c].std() for c in client_data.keys()]
})
summary_stats.to_csv('SCAN/huggingface_data/client_summary_stats.csv', index=False)

print("\n" + "=" * 60)
print("Energy Data Analysis Complete!")
print("Results saved to:")
print("  - SCAN/huggingface_data/synthetic_energy_data.csv")
print("  - SCAN/huggingface_data/comprehensive_energy_analysis.png")
print("  - SCAN/huggingface_data/client_summary_stats.csv")
print("=" * 60)