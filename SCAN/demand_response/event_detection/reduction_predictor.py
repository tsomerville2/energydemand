"""
Demand Response Reduction Predictor
Uses ML models to predict available load reduction capacity
Implements event detection and reduction forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DEMAND RESPONSE REDUCTION PREDICTOR")
print("=" * 60)

class ReductionPredictor:
    """Predict available load reduction capacity using ML"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.feature_importance = {}
        
    def prepare_features(self, load_data, weather_data, timestamp):
        """Extract features for prediction"""
        features = {
            'hour': timestamp.hour,
            'day_of_week': timestamp.dayofweek,
            'month': timestamp.month,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'temperature': weather_data.loc[timestamp] if timestamp in weather_data.index else 75,
            'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            'day_sin': np.sin(2 * np.pi * timestamp.dayofweek / 7),
            'day_cos': np.cos(2 * np.pi * timestamp.dayofweek / 7)
        }
        
        # Add historical load features
        if timestamp in load_data.index:
            # Previous hour load
            prev_hour = timestamp - timedelta(hours=1)
            if prev_hour in load_data.index:
                features['prev_hour_load'] = load_data.loc[prev_hour]
            else:
                features['prev_hour_load'] = load_data.mean()
            
            # Same hour yesterday
            yesterday = timestamp - timedelta(days=1)
            if yesterday in load_data.index:
                features['yesterday_same_hour'] = load_data.loc[yesterday]
            else:
                features['yesterday_same_hour'] = load_data.mean()
            
            # Same hour last week
            last_week = timestamp - timedelta(days=7)
            if last_week in load_data.index:
                features['last_week_same_hour'] = load_data.loc[last_week]
            else:
                features['last_week_same_hour'] = load_data.mean()
            
            # Rolling averages
            window_end = timestamp - timedelta(hours=1)
            window_start = timestamp - timedelta(hours=25)
            window_data = load_data.loc[window_start:window_end]
            if len(window_data) > 0:
                features['rolling_avg_24h'] = window_data.mean()
                features['rolling_std_24h'] = window_data.std()
                features['rolling_max_24h'] = window_data.max()
                features['rolling_min_24h'] = window_data.min()
            else:
                features['rolling_avg_24h'] = load_data.mean()
                features['rolling_std_24h'] = load_data.std()
                features['rolling_max_24h'] = load_data.max()
                features['rolling_min_24h'] = load_data.min()
        
        return features
    
    def train_models(self, training_data):
        """Train ML models on historical reduction data"""
        X = training_data.drop(['reduction_capacity', 'reduction_achieved'], axis=1)
        y_capacity = training_data['reduction_capacity']
        y_achieved = training_data['reduction_achieved']
        
        # Split data
        X_train, X_test, y_cap_train, y_cap_test, y_ach_train, y_ach_test = train_test_split(
            X, y_capacity, y_achieved, test_size=0.2, random_state=42
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            # Train for reduction capacity
            model_capacity = model.__class__(**model.get_params())
            model_capacity.fit(X_train, y_cap_train)
            y_cap_pred = model_capacity.predict(X_test)
            
            # Train for reduction achieved
            model_achieved = model.__class__(**model.get_params())
            model_achieved.fit(X_train, y_ach_train)
            y_ach_pred = model_achieved.predict(X_test)
            
            # Store models
            self.trained_models[f"{model_name}_capacity"] = model_capacity
            self.trained_models[f"{model_name}_achieved"] = model_achieved
            
            # Calculate metrics
            results[model_name] = {
                'capacity_mae': mean_absolute_error(y_cap_test, y_cap_pred),
                'capacity_r2': r2_score(y_cap_test, y_cap_pred),
                'achieved_mae': mean_absolute_error(y_ach_test, y_ach_pred),
                'achieved_r2': r2_score(y_ach_test, y_ach_pred)
            }
            
            # Store feature importance
            if hasattr(model_capacity, 'feature_importances_'):
                self.feature_importance[model_name] = dict(zip(
                    X.columns, model_capacity.feature_importances_
                ))
        
        return results
    
    def predict_reduction(self, features_df, model_name='random_forest'):
        """Predict available reduction capacity and likely achievement"""
        capacity_model = self.trained_models.get(f"{model_name}_capacity")
        achieved_model = self.trained_models.get(f"{model_name}_achieved")
        
        if capacity_model and achieved_model:
            capacity_pred = capacity_model.predict(features_df)
            achieved_pred = achieved_model.predict(features_df)
            return capacity_pred[0], achieved_pred[0]
        return None, None

class DemandResponseEventDetector:
    """Detect and respond to demand response events"""
    
    def __init__(self):
        self.event_triggers = {
            'price_spike': 80,  # $/MWh threshold
            'grid_stress': 0.95,  # 95% of capacity
            'emergency': 'manual'
        }
        
    def detect_event(self, current_price, grid_utilization, manual_trigger=False):
        """Detect if DR event should be triggered"""
        event_detected = False
        event_type = None
        event_severity = 'normal'
        
        if manual_trigger:
            event_detected = True
            event_type = 'emergency'
            event_severity = 'critical'
        elif current_price > self.event_triggers['price_spike']:
            event_detected = True
            event_type = 'price_spike'
            event_severity = 'high' if current_price > 100 else 'medium'
        elif grid_utilization > self.event_triggers['grid_stress']:
            event_detected = True
            event_type = 'grid_stress'
            event_severity = 'high'
        
        return {
            'event_detected': event_detected,
            'event_type': event_type,
            'severity': event_severity,
            'timestamp': datetime.now()
        }

# Generate training data
print("\n1. Generating Training Data...")
print("-" * 40)

np.random.seed(42)

# Create historical data (3 months)
start_date = datetime(2024, 6, 1)
end_date = datetime(2024, 8, 31, 23, 0, 0)
timestamps = pd.date_range(start_date, end_date, freq='h')

# Generate load and weather data
load_data = pd.Series(
    30 + 10 * np.sin(np.arange(len(timestamps)) * 2 * np.pi / 24) +
    5 * np.sin(np.arange(len(timestamps)) * 2 * np.pi / (24 * 7)) +
    np.random.normal(0, 3, len(timestamps)),
    index=timestamps,
    name='load_kw'
)

weather_data = pd.Series(
    75 + 15 * np.sin(np.arange(len(timestamps)) * 2 * np.pi / (24 * 30)) +
    np.random.normal(0, 5, len(timestamps)),
    index=timestamps,
    name='temperature'
)

# Generate training samples with reduction data
predictor = ReductionPredictor()
training_samples = []

for i in range(500):  # Generate 500 training samples
    # Random timestamp
    idx = np.random.randint(24, len(timestamps) - 24)
    ts = timestamps[idx]
    
    # Extract features
    features = predictor.prepare_features(load_data, weather_data, ts)
    
    # Simulate reduction capacity (based on time of day and load)
    base_load = load_data.loc[ts]
    if 9 <= ts.hour <= 17 and ts.weekday() < 5:  # Business hours
        reduction_capacity = base_load * np.random.uniform(0.15, 0.35)  # 15-35% reduction possible
    else:
        reduction_capacity = base_load * np.random.uniform(0.05, 0.15)  # 5-15% reduction possible
    
    # Simulate achieved reduction (usually 70-95% of capacity)
    achievement_rate = np.random.uniform(0.7, 0.95)
    reduction_achieved = reduction_capacity * achievement_rate
    
    # Add to training data
    sample = features.copy()
    sample['reduction_capacity'] = reduction_capacity
    sample['reduction_achieved'] = reduction_achieved
    training_samples.append(sample)

training_df = pd.DataFrame(training_samples)
print(f"Generated {len(training_df)} training samples")
print(f"Average reduction capacity: {training_df['reduction_capacity'].mean():.2f} kW")
print(f"Average reduction achieved: {training_df['reduction_achieved'].mean():.2f} kW")

# Train models
print("\n2. Training ML Models...")
print("-" * 40)

results = predictor.train_models(training_df)

for model_name, metrics in results.items():
    print(f"\n{model_name.replace('_', ' ').title()}:")
    print(f"  Capacity Prediction - MAE: {metrics['capacity_mae']:.2f} kW, R²: {metrics['capacity_r2']:.3f}")
    print(f"  Achieved Prediction - MAE: {metrics['achieved_mae']:.2f} kW, R²: {metrics['achieved_r2']:.3f}")

# Event detection simulation
print("\n3. Simulating Demand Response Events...")
print("-" * 40)

detector = DemandResponseEventDetector()

# Simulate a day with potential events
test_date = datetime(2024, 9, 1)
test_hours = pd.date_range(test_date, test_date + timedelta(hours=23), freq='h')

events = []
predictions = []

for hour in test_hours:
    # Simulate market conditions
    if hour.hour in [14, 15, 16, 17]:  # Peak hours
        price = np.random.uniform(70, 120)
        grid_util = np.random.uniform(0.85, 0.98)
    else:
        price = np.random.uniform(30, 60)
        grid_util = np.random.uniform(0.60, 0.85)
    
    # Detect event
    event = detector.detect_event(price, grid_util)
    
    if event['event_detected']:
        # Get features for prediction
        features = predictor.prepare_features(load_data, weather_data, hour)
        # Ensure all training features are present
        for col in training_df.columns:
            if col not in ['reduction_capacity', 'reduction_achieved'] and col not in features:
                features[col] = 0  # Default value for missing features
        features_df = pd.DataFrame([features])
        # Reorder columns to match training
        feature_cols = [col for col in training_df.columns if col not in ['reduction_capacity', 'reduction_achieved']]
        features_df = features_df[feature_cols]
        
        # Predict reduction
        capacity, achieved = predictor.predict_reduction(features_df)
        
        event_data = {
            'hour': hour,
            'price': price,
            'grid_utilization': grid_util,
            'event_type': event['event_type'],
            'severity': event['severity'],
            'predicted_capacity': capacity,
            'predicted_achieved': achieved
        }
        events.append(event_data)
        
        print(f"Event at {hour.strftime('%H:%00')} - Type: {event['event_type']}, "
              f"Predicted reduction: {achieved:.1f} kW")

if not events:
    print("No events detected in simulation")

# Feature importance analysis
print("\n4. Feature Importance Analysis...")
print("-" * 40)

if 'random_forest' in predictor.feature_importance:
    importance = predictor.feature_importance['random_forest']
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top 10 most important features for reduction prediction:")
    for i, (feature, score) in enumerate(sorted_features, 1):
        print(f"  {i:2}. {feature:25} {score:.4f}")

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Training data distribution
ax = axes[0, 0]
ax.scatter(training_df['hour'], training_df['reduction_capacity'], alpha=0.5, s=10, c='blue', label='Capacity')
ax.scatter(training_df['hour'], training_df['reduction_achieved'], alpha=0.5, s=10, c='green', label='Achieved')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Reduction (kW)')
ax.set_title('Training Data: Reduction by Hour')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Model comparison
ax = axes[0, 1]
models = list(results.keys())
capacity_r2 = [results[m]['capacity_r2'] for m in models]
achieved_r2 = [results[m]['achieved_r2'] for m in models]

x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, capacity_r2, width, label='Capacity R²', color='steelblue')
ax.bar(x + width/2, achieved_r2, width, label='Achieved R²', color='green')
ax.set_xlabel('Model')
ax.set_ylabel('R² Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ').title() for m in models])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Feature importance
ax = axes[0, 2]
if sorted_features:
    features = [f[0] for f in sorted_features[:8]]
    importances = [f[1] for f in sorted_features[:8]]
    ax.barh(range(len(features)), importances, color='coral')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title('Top Feature Importance')
    ax.grid(True, alpha=0.3, axis='x')

# Plot 4: Predictions vs Actual (sample)
ax = axes[1, 0]
sample_size = 100
sample_indices = np.random.choice(len(training_df), sample_size, replace=False)
sample_X = training_df.drop(['reduction_capacity', 'reduction_achieved'], axis=1).iloc[sample_indices]
sample_y_true = training_df['reduction_achieved'].iloc[sample_indices]

model = predictor.trained_models.get('random_forest_achieved')
if model:
    sample_y_pred = model.predict(sample_X)
    ax.scatter(sample_y_true, sample_y_pred, alpha=0.5, s=20)
    ax.plot([sample_y_true.min(), sample_y_true.max()], 
            [sample_y_true.min(), sample_y_true.max()], 
            'r--', label='Perfect Prediction')
    ax.set_xlabel('True Reduction (kW)')
    ax.set_ylabel('Predicted Reduction (kW)')
    ax.set_title('Prediction vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 5: Reduction capacity by temperature
ax = axes[1, 1]
temp_bins = pd.cut(training_df['temperature'], bins=10)
capacity_by_temp = training_df.groupby(temp_bins)['reduction_capacity'].mean()
ax.bar(range(len(capacity_by_temp)), capacity_by_temp.values, color='orange', alpha=0.7)
ax.set_xlabel('Temperature Bin')
ax.set_ylabel('Average Capacity (kW)')
ax.set_title('Reduction Capacity vs Temperature')
ax.set_xticklabels([f"{int(b.left)}-{int(b.right)}" for b in capacity_by_temp.index], rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Event timeline
ax = axes[1, 2]
if events:
    event_df = pd.DataFrame(events)
    hours = [e['hour'].hour for e in events]
    reductions = [e['predicted_achieved'] for e in events]
    colors = ['red' if e['severity'] == 'high' else 'orange' if e['severity'] == 'medium' else 'yellow' 
              for e in events]
    
    ax.bar(hours, reductions, color=colors, alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Predicted Reduction (kW)')
    ax.set_title('Predicted DR Events - Test Day')
    ax.grid(True, alpha=0.3, axis='y')
else:
    ax.text(0.5, 0.5, 'No Events Detected', ha='center', va='center', fontsize=12)
    ax.set_title('Predicted DR Events - Test Day')

plt.tight_layout()
plt.savefig('SCAN/demand_response/event_detection/reduction_predictions.png', dpi=150, bbox_inches='tight')

# Save results
if events:
    event_df = pd.DataFrame(events)
    event_df.to_csv('SCAN/demand_response/event_detection/event_predictions.csv', index=False)

print("\n" + "=" * 60)
print("Reduction prediction complete!")
print("Results saved to:")
print("  - SCAN/demand_response/event_detection/reduction_predictions.png")
if events:
    print("  - SCAN/demand_response/event_detection/event_predictions.csv")
print("=" * 60)