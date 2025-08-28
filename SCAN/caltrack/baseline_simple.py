"""
Simplified CalTRACK-style Baseline Estimation
Implements core CalTRACK methodology using standard libraries
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Generate realistic building energy data
np.random.seed(42)

print("=" * 60)
print("CALTRACK-STYLE BASELINE ESTIMATION")
print("=" * 60)

# Date ranges
baseline_start = datetime(2022, 1, 1)
baseline_end = datetime(2022, 12, 31)
reporting_start = datetime(2023, 1, 1)
reporting_end = datetime(2023, 6, 30)

# Generate daily data
baseline_days = pd.date_range(baseline_start, baseline_end, freq='D')
reporting_days = pd.date_range(reporting_start, reporting_end, freq='D')

def generate_daily_temps(dates):
    """Generate realistic daily temperature data"""
    day_of_year = dates.dayofyear
    base_temp = 55 + 25 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
    noise = np.random.normal(0, 5, len(dates))
    return base_temp + noise

def calculate_hdd_cdd(temps, heating_bp=65, cooling_bp=65):
    """Calculate heating and cooling degree days"""
    hdd = np.maximum(heating_bp - temps, 0)
    cdd = np.maximum(temps - cooling_bp, 0)
    return hdd, cdd

# Generate baseline data
baseline_temps = generate_daily_temps(baseline_days)
baseline_hdd, baseline_cdd = calculate_hdd_cdd(baseline_temps)

# Generate consumption with CalTRACK-style model
base_load = 1200  # Daily base load in kWh
heating_coef = 15  # kWh per HDD
cooling_coef = 20  # kWh per CDD

baseline_consumption = (
    base_load + 
    heating_coef * baseline_hdd + 
    cooling_coef * baseline_cdd +
    np.random.normal(0, 50, len(baseline_days))
)
baseline_consumption = np.maximum(baseline_consumption, 100)

# Create baseline dataframe
baseline_df = pd.DataFrame({
    'date': baseline_days,
    'temperature': baseline_temps,
    'hdd': baseline_hdd,
    'cdd': baseline_cdd,
    'consumption': baseline_consumption
})

print("\nBaseline Period Statistics:")
print(f"  Date Range: {baseline_start.date()} to {baseline_end.date()}")
print(f"  Total Days: {len(baseline_df)}")
print(f"  Avg Daily Consumption: {baseline_df['consumption'].mean():.0f} kWh")
print(f"  Total Consumption: {baseline_df['consumption'].sum():,.0f} kWh")

# Fit CalTRACK-style model
print("\nFitting Baseline Model (CalTRACK Method)...")

# Prepare features
X_baseline = baseline_df[['hdd', 'cdd']].values
X_baseline_with_intercept = np.column_stack([np.ones(len(X_baseline)), X_baseline])
y_baseline = baseline_df['consumption'].values

# Fit model
model = LinearRegression()
model.fit(X_baseline, y_baseline)
baseline_predictions = model.predict(X_baseline)
r2 = r2_score(y_baseline, baseline_predictions)

print(f"\nBaseline Model Coefficients:")
print(f"  Base Load (Intercept): {model.intercept_:.0f} kWh/day")
print(f"  Heating Slope: {model.coef_[0]:.1f} kWh/HDD")
print(f"  Cooling Slope: {model.coef_[1]:.1f} kWh/CDD")
print(f"  R-squared: {r2:.3f}")
print(f"  CVRMSE: {np.sqrt(np.mean((y_baseline - baseline_predictions)**2)) / np.mean(y_baseline) * 100:.1f}%")

# Generate reporting period data (post-retrofit)
reporting_temps = generate_daily_temps(reporting_days)
reporting_hdd, reporting_cdd = calculate_hdd_cdd(reporting_temps)

# Simulate 20% efficiency improvement
reporting_consumption = (
    base_load * 0.85 +  # Reduced base load
    heating_coef * 0.8 * reporting_hdd +  # More efficient heating
    cooling_coef * 0.75 * reporting_cdd +  # More efficient cooling
    np.random.normal(0, 40, len(reporting_days))
)
reporting_consumption = np.maximum(reporting_consumption, 100)

# Create reporting dataframe
reporting_df = pd.DataFrame({
    'date': reporting_days,
    'temperature': reporting_temps,
    'hdd': reporting_hdd,
    'cdd': reporting_cdd,
    'actual_consumption': reporting_consumption
})

# Calculate counterfactual
print("\nCalculating Counterfactual (No Retrofit Scenario)...")
X_reporting = reporting_df[['hdd', 'cdd']].values
counterfactual = model.predict(X_reporting)
reporting_df['counterfactual'] = counterfactual

# Calculate savings
total_actual = reporting_df['actual_consumption'].sum()
total_counterfactual = reporting_df['counterfactual'].sum()
total_savings = total_counterfactual - total_actual
percent_savings = (total_savings / total_counterfactual) * 100

print(f"\nReporting Period Results:")
print(f"  Date Range: {reporting_start.date()} to {reporting_end.date()}")
print(f"  Total Days: {len(reporting_df)}")
print(f"  Actual Consumption: {total_actual:,.0f} kWh")
print(f"  Counterfactual (Baseline): {total_counterfactual:,.0f} kWh")
print(f"  Energy Savings: {total_savings:,.0f} kWh")
print(f"  Percent Savings: {percent_savings:.1f}%")

# Calculate uncertainty
residual_std = np.std(y_baseline - baseline_predictions)
confidence_interval = 1.96 * residual_std * np.sqrt(len(reporting_df))
print(f"  95% Confidence Interval: ±{confidence_interval:,.0f} kWh")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Baseline model fit
ax = axes[0, 0]
ax.scatter(baseline_temps, baseline_consumption, alpha=0.5, s=10, label='Actual')
ax.scatter(baseline_temps, baseline_predictions, alpha=0.5, s=10, color='red', label='Model')
ax.set_xlabel('Temperature (°F)')
ax.set_ylabel('Daily Consumption (kWh)')
ax.set_title('Baseline Period: Model Fit')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: HDD/CDD relationship
ax = axes[0, 1]
ax.scatter(baseline_hdd, baseline_consumption, alpha=0.5, s=10, label='HDD', color='blue')
ax.scatter(baseline_cdd, baseline_consumption, alpha=0.5, s=10, label='CDD', color='red')
ax.set_xlabel('Degree Days')
ax.set_ylabel('Daily Consumption (kWh)')
ax.set_title('Consumption vs Degree Days')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Residuals
ax = axes[0, 2]
residuals = baseline_consumption - baseline_predictions
ax.scatter(baseline_predictions, residuals, alpha=0.5, s=10)
ax.axhline(y=0, color='red', linestyle='--')
ax.set_xlabel('Predicted Consumption (kWh)')
ax.set_ylabel('Residuals (kWh)')
ax.set_title('Model Residuals')
ax.grid(True, alpha=0.3)

# Plot 4: Time series comparison
ax = axes[1, 0]
ax.plot(reporting_df['date'], reporting_df['actual_consumption'], 
        label='Actual', linewidth=1.5, alpha=0.8)
ax.plot(reporting_df['date'], reporting_df['counterfactual'],
        label='Counterfactual', linewidth=1.5, linestyle='--', alpha=0.8)
ax.fill_between(reporting_df['date'], 
                reporting_df['actual_consumption'],
                reporting_df['counterfactual'],
                alpha=0.2, color='green', label='Savings')
ax.set_xlabel('Date')
ax.set_ylabel('Daily Consumption (kWh)')
ax.set_title('Actual vs Counterfactual')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Cumulative savings
ax = axes[1, 1]
daily_savings = reporting_df['counterfactual'] - reporting_df['actual_consumption']
cumulative_savings = daily_savings.cumsum()
ax.plot(reporting_df['date'], cumulative_savings, linewidth=2, color='green')
ax.fill_between(reporting_df['date'], 0, cumulative_savings, alpha=0.3, color='green')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Savings (kWh)')
ax.set_title(f'Total Savings: {total_savings:,.0f} kWh')
ax.grid(True, alpha=0.3)

# Plot 6: Monthly savings breakdown
ax = axes[1, 2]
reporting_df['month'] = pd.to_datetime(reporting_df['date']).dt.to_period('M')
monthly_savings = reporting_df.groupby('month').apply(
    lambda x: (x['counterfactual'] - x['actual_consumption']).sum()
)
months = [str(m) for m in monthly_savings.index]
ax.bar(range(len(months)), monthly_savings.values, color='green', alpha=0.7)
ax.set_xticks(range(len(months)))
ax.set_xticklabels(months, rotation=45)
ax.set_xlabel('Month')
ax.set_ylabel('Monthly Savings (kWh)')
ax.set_title('Monthly Savings Breakdown')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SCAN/caltrack/baseline_results.png', dpi=150, bbox_inches='tight')

# Save results to CSV
reporting_df['daily_savings'] = daily_savings
reporting_df['cumulative_savings'] = cumulative_savings
reporting_df.to_csv('SCAN/caltrack/baseline_analysis.csv', index=False)

print("\n" + "=" * 60)
print("CalTRACK-Style Analysis Complete!")
print("Results saved to:")
print("  - SCAN/caltrack/baseline_results.png")
print("  - SCAN/caltrack/baseline_analysis.csv")
print("=" * 60)