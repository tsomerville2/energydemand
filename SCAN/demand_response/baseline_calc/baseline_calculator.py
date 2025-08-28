"""
Demand Response Baseline Calculator
Implements CAISO 10-in-10 baseline methodology and other methods
Real implementation for calculating customer baseline load (CBL)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DEMAND RESPONSE BASELINE CALCULATOR")
print("=" * 60)

class BaselineCalculator:
    """Calculate customer baseline load using various methodologies"""
    
    def __init__(self):
        self.methods = {
            'CAISO_10_in_10': self.calculate_caiso_10_in_10,
            'CAISO_5_in_10': self.calculate_caiso_5_in_10,
            'Weather_Matching': self.calculate_weather_matching,
            'Regression': self.calculate_regression_baseline
        }
        
    def calculate_caiso_10_in_10(self, load_data, event_date, event_hours):
        """
        CAISO 10-in-10 baseline calculation
        - Look back for 10 similar non-event days
        - Average their consumption
        - Apply load point adjustment (up to 20%)
        """
        # Find eligible baseline days (non-event weekdays in past 45 days)
        eligible_days = []
        current_date = event_date - timedelta(days=1)
        lookback_limit = event_date - timedelta(days=45)
        
        while len(eligible_days) < 10 and current_date >= lookback_limit:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
                # Check if it's not an event day (simplified - no prior events)
                if current_date in load_data.index:
                    eligible_days.append(current_date)
            current_date -= timedelta(days=1)
        
        if len(eligible_days) < 10:
            print(f"Warning: Only found {len(eligible_days)} eligible days")
        
        # Calculate baseline as average of eligible days
        baseline_profiles = []
        for day in eligible_days:
            day_profile = load_data.loc[day:day + timedelta(hours=23)]
            if len(day_profile) == 24:
                baseline_profiles.append(day_profile.values)
        
        if not baseline_profiles:
            return None
        
        # Average the profiles
        baseline = np.mean(baseline_profiles, axis=0)
        
        # Apply load point adjustment (using 3 hours before event)
        if event_hours[0] >= 4:
            adjustment_hours = range(event_hours[0] - 4, event_hours[0] - 1)
            actual_pre_event = load_data.loc[event_date + timedelta(hours=adjustment_hours[0]):
                                            event_date + timedelta(hours=adjustment_hours[-1])].mean()
            baseline_pre_event = np.mean(baseline[adjustment_hours])
            
            # Calculate adjustment factor (capped at ±20%)
            adjustment_factor = actual_pre_event / baseline_pre_event if baseline_pre_event > 0 else 1.0
            adjustment_factor = max(0.8, min(1.2, adjustment_factor))
            
            # Apply adjustment
            baseline = baseline * adjustment_factor
        
        return baseline
    
    def calculate_caiso_5_in_10(self, load_data, event_date, event_hours):
        """
        CAISO 5-in-10 baseline (for residential)
        Select 5 highest consumption days from last 10 eligible days
        """
        # Get 10 eligible days first
        eligible_days = []
        current_date = event_date - timedelta(days=1)
        lookback_limit = event_date - timedelta(days=45)
        
        while len(eligible_days) < 10 and current_date >= lookback_limit:
            if current_date.weekday() < 5:
                if current_date in load_data.index:
                    eligible_days.append(current_date)
            current_date -= timedelta(days=1)
        
        # Calculate total consumption for each day and select top 5
        day_consumptions = []
        for day in eligible_days:
            day_total = load_data.loc[day:day + timedelta(hours=23)].sum()
            day_consumptions.append((day, day_total))
        
        # Sort and take top 5
        day_consumptions.sort(key=lambda x: x[1], reverse=True)
        top_5_days = [day for day, _ in day_consumptions[:5]]
        
        # Calculate baseline from top 5 days
        baseline_profiles = []
        for day in top_5_days:
            day_profile = load_data.loc[day:day + timedelta(hours=23)]
            if len(day_profile) == 24:
                baseline_profiles.append(day_profile.values)
        
        return np.mean(baseline_profiles, axis=0) if baseline_profiles else None
    
    def calculate_weather_matching(self, load_data, weather_data, event_date, event_hours):
        """
        Weather-based baseline matching
        Find days with similar weather conditions
        """
        event_weather = weather_data.loc[event_date]
        
        # Find 5 days with most similar weather
        weather_differences = []
        for date in load_data.index.normalize().unique():
            if date != event_date and date < event_date:
                if date in weather_data.index:
                    diff = abs(weather_data.loc[date] - event_weather)
                    weather_differences.append((date, diff))
        
        # Sort by weather similarity
        weather_differences.sort(key=lambda x: x[1])
        similar_days = [day for day, _ in weather_differences[:5]]
        
        # Calculate baseline from similar weather days
        baseline_profiles = []
        for day in similar_days:
            day_profile = load_data.loc[day:day + timedelta(hours=23)]
            if len(day_profile) == 24:
                baseline_profiles.append(day_profile.values)
        
        return np.mean(baseline_profiles, axis=0) if baseline_profiles else None
    
    def calculate_regression_baseline(self, load_data, features, event_date):
        """
        Regression-based baseline using historical patterns
        """
        from sklearn.linear_model import LinearRegression
        
        # Prepare training data (exclude event day)
        train_data = load_data[load_data.index.date != event_date]
        
        # Simple features: hour of day, day of week, temperature
        X_train = features[features.index.isin(train_data.index)]
        y_train = train_data[train_data.index.isin(X_train.index)]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict baseline for event day
        event_features = features[features.index.date == event_date]
        baseline = model.predict(event_features)
        
        return baseline

# Generate sample building load data
print("\n1. Generating Sample Building Load Data...")
print("-" * 40)

np.random.seed(42)

# Create hourly timestamps for 60 days
start_date = datetime(2024, 7, 1)
end_date = datetime(2024, 8, 30, 23, 0, 0)
timestamps = pd.date_range(start_date, end_date, freq='h')

# Generate realistic commercial building load pattern
def generate_building_load(timestamps):
    """Generate realistic commercial building load profile"""
    loads = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        
        # Base load varies by time and day
        if day_of_week < 5:  # Weekday
            if 6 <= hour <= 8:
                base_load = 35  # Morning ramp-up
            elif 9 <= hour <= 17:
                base_load = 40  # Business hours
            elif 18 <= hour <= 20:
                base_load = 35  # Evening ramp-down
            else:
                base_load = 25  # Night/early morning
        else:  # Weekend
            base_load = 20
        
        # Add temperature effect (summer cooling)
        temp_effect = 0
        if 10 <= hour <= 18:
            temp_effect = 5 * np.sin((hour - 10) * np.pi / 8)
        
        # Add random variation
        variation = np.random.normal(0, 2)
        
        load = base_load + temp_effect + variation
        loads.append(max(15, load))  # Minimum 15 kW
    
    return loads

# Create load data
load_values = generate_building_load(timestamps)
load_data = pd.Series(load_values, index=timestamps, name='load_kw')

print(f"Generated {len(load_data)} hours of building load data")
print(f"Average load: {load_data.mean():.1f} kW")
print(f"Peak load: {load_data.max():.1f} kW")

# Create weather data (temperature)
weather_data = pd.Series(
    70 + 15 * np.sin((np.arange(len(timestamps)) % (24 * 30)) * 2 * np.pi / (24 * 30)) + 
    np.random.normal(0, 3, len(timestamps)),
    index=timestamps,
    name='temperature'
)

# Define demand response event
event_date = datetime(2024, 8, 15)
event_hours = [14, 15, 16, 17]  # 2 PM to 6 PM
print(f"\nDemand Response Event:")
print(f"  Date: {event_date.date()}")
print(f"  Hours: {event_hours[0]}:00 - {event_hours[-1]+1}:00")

# Calculate baselines using different methods
calculator = BaselineCalculator()

print("\n2. Calculating Baselines...")
print("-" * 40)

# CAISO 10-in-10
baseline_10_10 = calculator.calculate_caiso_10_in_10(load_data, event_date, event_hours)
print(f"CAISO 10-in-10 baseline calculated: {baseline_10_10 is not None}")
if baseline_10_10 is not None:
    print(f"  Average baseline during event: {np.mean(baseline_10_10[event_hours]):.1f} kW")

# CAISO 5-in-10
baseline_5_10 = calculator.calculate_caiso_5_in_10(load_data, event_date, event_hours)
print(f"CAISO 5-in-10 baseline calculated: {baseline_5_10 is not None}")
if baseline_5_10 is not None:
    print(f"  Average baseline during event: {np.mean(baseline_5_10[event_hours]):.1f} kW")

# Weather matching
baseline_weather = calculator.calculate_weather_matching(load_data, weather_data, event_date, event_hours)
print(f"Weather matching baseline calculated: {baseline_weather is not None}")

# Get actual load for event day
actual_load = load_data.loc[event_date:event_date + timedelta(hours=23)].values

# Simulate demand response (reduce load during event hours)
reduced_load = actual_load.copy()
for hour in event_hours:
    reduced_load[hour] = actual_load[hour] * 0.75  # 25% reduction

print("\n3. Demand Response Performance:")
print("-" * 40)

# Calculate reduction
baseline_event = baseline_10_10[event_hours] if baseline_10_10 is not None else actual_load[event_hours]
actual_event = reduced_load[event_hours]
reduction = baseline_event - actual_event
total_reduction = np.sum(reduction)
avg_reduction_pct = np.mean(reduction / baseline_event) * 100

print(f"Baseline during event: {baseline_event.mean():.1f} kW average")
print(f"Actual during event: {actual_event.mean():.1f} kW average")
print(f"Total reduction: {total_reduction:.1f} kWh")
print(f"Average reduction: {avg_reduction_pct:.1f}%")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Baseline comparison
ax = axes[0, 0]
hours = range(24)
ax.plot(hours, actual_load, label='Actual (No DR)', linewidth=2, alpha=0.7)
ax.plot(hours, reduced_load, label='Actual (With DR)', linewidth=2)
if baseline_10_10 is not None:
    ax.plot(hours, baseline_10_10, label='CAISO 10-in-10', linestyle='--', linewidth=2)
if baseline_5_10 is not None:
    ax.plot(hours, baseline_5_10, label='CAISO 5-in-10', linestyle=':', linewidth=2)

# Highlight event hours
for hour in event_hours:
    ax.axvspan(hour, hour+1, alpha=0.2, color='red')

ax.set_xlabel('Hour of Day')
ax.set_ylabel('Load (kW)')
ax.set_title(f'Demand Response Event - {event_date.date()}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Reduction during event
ax = axes[0, 1]
event_hour_labels = [f"{h}:00" for h in event_hours]
ax.bar(event_hour_labels, baseline_event, alpha=0.5, label='Baseline', color='blue')
ax.bar(event_hour_labels, actual_event, alpha=0.7, label='Actual', color='green')
ax.bar(event_hour_labels, reduction, alpha=0.7, label='Reduction', color='red', bottom=actual_event)
ax.set_xlabel('Hour')
ax.set_ylabel('Load (kW)')
ax.set_title('Hourly Load Reduction During Event')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Historical baselines
ax = axes[1, 0]
# Show last 10 days used for baseline
lookback_days = 10
for i in range(lookback_days):
    day = event_date - timedelta(days=i+1)
    if day.weekday() < 5:  # Weekday
        day_load = load_data.loc[day:day + timedelta(hours=23)]
        if len(day_load) == 24:
            ax.plot(hours, day_load.values, alpha=0.3, color='gray')

ax.plot(hours, baseline_10_10, label='10-in-10 Average', linewidth=2, color='blue')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Load (kW)')
ax.set_title('Historical Days Used for Baseline')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Accuracy metrics
ax = axes[1, 1]
methods = ['10-in-10', '5-in-10']
baselines = [baseline_10_10, baseline_5_10]
rmse_values = []
mape_values = []

for baseline in baselines:
    if baseline is not None:
        # Calculate RMSE and MAPE (using non-event hours)
        non_event_hours = [h for h in range(24) if h not in event_hours]
        rmse = np.sqrt(np.mean((actual_load[non_event_hours] - baseline[non_event_hours])**2))
        mape = np.mean(np.abs((actual_load[non_event_hours] - baseline[non_event_hours]) / 
                             actual_load[non_event_hours])) * 100
        rmse_values.append(rmse)
        mape_values.append(mape)
    else:
        rmse_values.append(0)
        mape_values.append(0)

x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, rmse_values, width, label='RMSE (kW)', color='steelblue')
ax2 = ax.twinx()
ax2.bar(x + width/2, mape_values, width, label='MAPE (%)', color='orange')

ax.set_xlabel('Baseline Method')
ax.set_ylabel('RMSE (kW)', color='steelblue')
ax2.set_ylabel('MAPE (%)', color='orange')
ax.set_title('Baseline Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='orange')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SCAN/demand_response/baseline_calc/baseline_analysis.png', dpi=150, bbox_inches='tight')

# Save data to CSV
results_df = pd.DataFrame({
    'hour': range(24),
    'actual_no_dr': actual_load,
    'actual_with_dr': reduced_load,
    'baseline_10_10': baseline_10_10 if baseline_10_10 is not None else [np.nan]*24,
    'baseline_5_10': baseline_5_10 if baseline_5_10 is not None else [np.nan]*24,
    'reduction': [reduction[event_hours.index(h)] if h in event_hours else 0 for h in range(24)]
})
results_df.to_csv('SCAN/demand_response/baseline_calc/baseline_results.csv', index=False)

print("\n" + "=" * 60)
print("Baseline calculation complete!")
print("Results saved to:")
print("  - SCAN/demand_response/baseline_calc/baseline_analysis.png")
print("  - SCAN/demand_response/baseline_calc/baseline_results.csv")
print("=" * 60)