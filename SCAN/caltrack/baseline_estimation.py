"""
CalTRACK/OpenEEmeter Baseline Estimation Demo
Real implementation using synthetic building energy data
Demonstrates weather-normalized baseline creation and savings calculation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import eemeter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic but realistic building energy consumption data
np.random.seed(42)

# Create date range for baseline period (1 year) and reporting period (6 months)
baseline_start = datetime(2022, 1, 1)
baseline_end = datetime(2022, 12, 31)
reporting_start = datetime(2023, 1, 1)
reporting_end = datetime(2023, 6, 30)

# Generate hourly timestamps
baseline_index = pd.date_range(start=baseline_start, end=baseline_end, freq='h')
reporting_index = pd.date_range(start=reporting_start, end=reporting_end, freq='h')

# Create temperature data (realistic seasonal pattern)
def generate_temperature(dates):
    """Generate realistic temperature data with seasonal patterns"""
    day_of_year = dates.dayofyear
    # Base temperature curve (seasonal)
    base_temp = 55 + 25 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
    # Daily variation
    hour_of_day = dates.hour
    daily_variation = 5 * np.sin((hour_of_day - 6) * 2 * np.pi / 24)
    # Random noise
    noise = np.random.normal(0, 3, len(dates))
    return base_temp + daily_variation + noise

# Generate baseline consumption (before energy efficiency measures)
def generate_consumption(temps, base_load=50, heating_slope=2, cooling_slope=1.5):
    """Generate consumption based on temperature"""
    consumption = np.ones(len(temps)) * base_load
    # Heating (when temp < 60°F)
    heating_mask = temps < 60
    consumption[heating_mask] += heating_slope * (60 - temps[heating_mask])
    # Cooling (when temp > 70°F)
    cooling_mask = temps > 70
    consumption[cooling_mask] += cooling_slope * (temps[cooling_mask] - 70)
    # Add random variation
    consumption += np.random.normal(0, 5, len(temps))
    return np.maximum(consumption, 10)  # Ensure positive values

# Generate data
baseline_temps = generate_temperature(baseline_index)
baseline_consumption = generate_consumption(baseline_temps)

# For reporting period, simulate 15% efficiency improvement
reporting_temps = generate_temperature(reporting_index)
reporting_consumption = generate_consumption(
    reporting_temps, 
    base_load=45,  # Reduced base load
    heating_slope=1.7,  # More efficient heating
    cooling_slope=1.2   # More efficient cooling
) * 0.85  # Overall 15% reduction

# Create meter data DataFrames
baseline_meter_data = pd.DataFrame({
    'start': baseline_index[:-1],
    'value': baseline_consumption[:-1],
    'estimated': False
})

reporting_meter_data = pd.DataFrame({
    'start': reporting_index[:-1],
    'value': reporting_consumption[:-1],
    'estimated': False
})

# Create temperature data DataFrames
baseline_temperature_data = pd.DataFrame({
    'date': baseline_index,
    'temperature': baseline_temps
})

reporting_temperature_data = pd.DataFrame({
    'date': reporting_index,
    'temperature': reporting_temps
})

# Convert to daily data for CalTRACK (as per standard practice)
baseline_daily = baseline_meter_data.set_index('start').resample('D').agg({
    'value': 'sum',
    'estimated': 'any'
}).reset_index()

reporting_daily = reporting_meter_data.set_index('start').resample('D').agg({
    'value': 'sum',
    'estimated': 'any'
}).reset_index()

baseline_temp_daily = baseline_temperature_data.set_index('date').resample('D').agg({
    'temperature': 'mean'
}).reset_index()

reporting_temp_daily = reporting_temperature_data.set_index('date').resample('D').agg({
    'temperature': 'mean'
}).reset_index()

print("=" * 60)
print("CALTRACK/OPENEEMETER BASELINE ESTIMATION")
print("=" * 60)

# Create CalTRACK baseline model
print("\nCreating baseline model from historical data...")

# Prepare data in eemeter format
baseline_data = eemeter.create_caltrack_daily_design_matrix(
    baseline_daily,
    baseline_temp_daily
)

# Fit baseline model
baseline_model = eemeter.CaltrackDailyModel()
baseline_model.fit(baseline_data)

print(f"\nBaseline Model Parameters:")
print(f"  Heating Balance Point: {baseline_model.model.parameters.loc[0, 'heating_bp']:.1f}°F")
print(f"  Cooling Balance Point: {baseline_model.model.parameters.loc[0, 'cooling_bp']:.1f}°F")
print(f"  Heating Slope: {baseline_model.model.parameters.loc[0, 'heating_slope']:.2f} kWh/°F")
print(f"  Cooling Slope: {baseline_model.model.parameters.loc[0, 'cooling_slope']:.2f} kWh/°F")
print(f"  Base Load: {baseline_model.model.parameters.loc[0, 'intercept']:.1f} kWh/day")
print(f"  R-squared: {baseline_model.model.parameters.loc[0, 'r_squared']:.3f}")

# Calculate counterfactual (what consumption would have been without efficiency measures)
print("\nCalculating counterfactual consumption...")
reporting_data = eemeter.create_caltrack_daily_design_matrix(
    reporting_daily,
    reporting_temp_daily
)

counterfactual = baseline_model.predict(reporting_data)

# Calculate savings
actual_consumption = reporting_daily['value'].sum()
predicted_consumption = counterfactual['predicted_usage'].sum()
savings = predicted_consumption - actual_consumption
savings_percentage = (savings / predicted_consumption) * 100

print(f"\nEnergy Savings Analysis:")
print(f"  Baseline Period: {baseline_start.strftime('%Y-%m-%d')} to {baseline_end.strftime('%Y-%m-%d')}")
print(f"  Reporting Period: {reporting_start.strftime('%Y-%m-%d')} to {reporting_end.strftime('%Y-%m-%d')}")
print(f"  Actual Consumption: {actual_consumption:,.0f} kWh")
print(f"  Predicted (Counterfactual): {predicted_consumption:,.0f} kWh")
print(f"  Energy Savings: {savings:,.0f} kWh ({savings_percentage:.1f}%)")

# Calculate uncertainty bounds
confidence_interval = 1.96 * np.sqrt(counterfactual['predicted_usage_variance'].sum())
print(f"  95% Confidence Interval: ±{confidence_interval:,.0f} kWh")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Baseline period - actual vs model
ax = axes[0, 0]
baseline_predictions = baseline_model.predict(baseline_data)
ax.scatter(baseline_temp_daily['temperature'], baseline_daily['value'], 
          alpha=0.5, s=20, label='Actual')
ax.scatter(baseline_temp_daily['temperature'], baseline_predictions['predicted_usage'],
          alpha=0.5, s=20, color='red', label='Model')
ax.set_xlabel('Daily Average Temperature (°F)')
ax.set_ylabel('Daily Consumption (kWh)')
ax.set_title('Baseline Period: Model Fit')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Time series comparison
ax = axes[0, 1]
ax.plot(reporting_daily['start'], reporting_daily['value'], 
        label='Actual (Post-Retrofit)', linewidth=2)
ax.plot(reporting_daily['start'], counterfactual['predicted_usage'],
        label='Counterfactual (Without Retrofit)', linewidth=2, linestyle='--')
ax.fill_between(reporting_daily['start'],
                counterfactual['predicted_usage'] - confidence_interval/len(counterfactual),
                counterfactual['predicted_usage'] + confidence_interval/len(counterfactual),
                alpha=0.2, label='95% CI')
ax.set_xlabel('Date')
ax.set_ylabel('Daily Consumption (kWh)')
ax.set_title('Reporting Period: Actual vs Counterfactual')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Cumulative savings
ax = axes[1, 0]
daily_savings = counterfactual['predicted_usage'].values - reporting_daily['value'].values
cumulative_savings = np.cumsum(daily_savings)
ax.plot(reporting_daily['start'], cumulative_savings, linewidth=2, color='green')
ax.fill_between(reporting_daily['start'], 0, cumulative_savings, alpha=0.3, color='green')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Savings (kWh)')
ax.set_title('Cumulative Energy Savings Over Time')
ax.grid(True, alpha=0.3)

# Plot 4: Temperature vs savings
ax = axes[1, 1]
ax.scatter(reporting_temp_daily['temperature'], daily_savings, alpha=0.5)
z = np.polyfit(reporting_temp_daily['temperature'], daily_savings, 2)
p = np.poly1d(z)
temps_sorted = np.sort(reporting_temp_daily['temperature'])
ax.plot(temps_sorted, p(temps_sorted), 'r-', linewidth=2, label='Trend')
ax.set_xlabel('Daily Average Temperature (°F)')
ax.set_ylabel('Daily Savings (kWh)')
ax.set_title('Savings vs Temperature')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SCAN/caltrack/baseline_estimation_results.png', dpi=150, bbox_inches='tight')

# Save detailed results
results_df = pd.DataFrame({
    'date': reporting_daily['start'],
    'actual_consumption_kwh': reporting_daily['value'],
    'counterfactual_kwh': counterfactual['predicted_usage'].values,
    'daily_savings_kwh': daily_savings,
    'cumulative_savings_kwh': cumulative_savings,
    'temperature_f': reporting_temp_daily['temperature'].values[:len(reporting_daily)]
})
results_df.to_csv('SCAN/caltrack/savings_analysis.csv', index=False)

print("\n" + "=" * 60)
print("Results saved to:")
print("  - SCAN/caltrack/baseline_estimation_results.png")
print("  - SCAN/caltrack/savings_analysis.csv")
print("=" * 60)