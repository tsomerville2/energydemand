"""
Demand Response Payment Calculator
Calculates payments based on actual load reductions
Implements various payment structures and incentive programs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DEMAND RESPONSE PAYMENT CALCULATOR")
print("=" * 60)

class PaymentCalculator:
    """Calculate DR payments based on reduction performance"""
    
    def __init__(self):
        self.payment_structures = {
            'capacity': {
                'name': 'Capacity Payment',
                'rate': 50,  # $/kW-month
                'description': 'Monthly payment for enrolled capacity'
            },
            'energy': {
                'name': 'Energy Payment',
                'rate': 0.15,  # $/kWh
                'description': 'Payment per kWh reduced during events'
            },
            'performance': {
                'name': 'Performance Incentive',
                'rate': 0.05,  # $/kWh bonus
                'description': 'Bonus for achieving >90% of committed reduction'
            },
            'peak': {
                'name': 'Peak Reduction Bonus',
                'rate': 0.25,  # $/kWh during peak
                'description': 'Additional payment during system peak hours'
            }
        }
        
        self.penalties = {
            'underperformance': 0.1,  # 10% penalty per kW short
            'non_response': 50,  # $50 penalty for not responding
            'false_baseline': 100  # $100 penalty for gaming baseline
        }
        
    def calculate_baseline(self, historical_load, event_date, event_hours):
        """Calculate baseline using CAISO 10-in-10 method"""
        # Simplified baseline calculation
        baseline_days = []
        for i in range(1, 11):
            day = event_date - timedelta(days=i)
            if day.weekday() < 5:  # Weekday
                baseline_days.append(day)
        
        if len(baseline_days) < 10:
            baseline_days = baseline_days[:5]  # Use available days
        
        # Calculate average load for event hours from baseline days
        baseline_loads = []
        for day in baseline_days:
            day_loads = []
            for hour in event_hours:
                timestamp = day.replace(hour=hour)
                if timestamp in historical_load.index:
                    day_loads.append(historical_load.loc[timestamp])
            if day_loads:
                baseline_loads.append(np.mean(day_loads))
        
        return np.mean(baseline_loads) if baseline_loads else 40  # Default 40 kW
    
    def calculate_reduction(self, baseline, actual, event_hours):
        """Calculate actual reduction achieved"""
        reductions = []
        for hour in event_hours:
            reduction = baseline - actual[hour]
            reductions.append(max(0, reduction))  # No negative reductions
        return reductions
    
    def calculate_payment(self, reduction_data, payment_type='comprehensive'):
        """Calculate payment based on reduction performance"""
        payments = {
            'capacity': 0,
            'energy': 0,
            'performance': 0,
            'peak': 0,
            'penalties': 0,
            'total': 0
        }
        
        # Capacity payment (monthly)
        if reduction_data['enrolled_capacity'] > 0:
            payments['capacity'] = (
                reduction_data['enrolled_capacity'] * 
                self.payment_structures['capacity']['rate']
            )
        
        # Energy payment (per event)
        total_reduction_kwh = sum(reduction_data['hourly_reductions'])
        payments['energy'] = (
            total_reduction_kwh * 
            self.payment_structures['energy']['rate']
        )
        
        # Performance incentive
        committed = reduction_data['committed_reduction'] * len(reduction_data['event_hours'])
        achieved = total_reduction_kwh
        performance_ratio = achieved / committed if committed > 0 else 0
        
        if performance_ratio >= 0.9:
            payments['performance'] = (
                total_reduction_kwh * 
                self.payment_structures['performance']['rate']
            )
        elif performance_ratio < 0.5:
            # Underperformance penalty
            shortfall = committed - achieved
            payments['penalties'] = -shortfall * self.penalties['underperformance']
        
        # Peak hour bonus
        peak_hours = [14, 15, 16, 17, 18]  # 2 PM - 7 PM
        peak_reductions = sum([
            reduction_data['hourly_reductions'][i] 
            for i, hour in enumerate(reduction_data['event_hours'])
            if hour in peak_hours
        ])
        payments['peak'] = (
            peak_reductions * 
            self.payment_structures['peak']['rate']
        )
        
        # Calculate total
        payments['total'] = (
            payments['capacity'] + 
            payments['energy'] + 
            payments['performance'] + 
            payments['peak'] + 
            payments['penalties']
        )
        
        return payments
    
    def generate_invoice(self, customer_info, reduction_data, payments):
        """Generate detailed invoice for DR participation"""
        invoice = {
            'invoice_number': f"DR-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}",
            'invoice_date': datetime.now().isoformat(),
            'customer': customer_info,
            'event_details': {
                'event_date': reduction_data['event_date'],
                'event_hours': reduction_data['event_hours'],
                'baseline_load': reduction_data['baseline'],
                'average_actual_load': np.mean(reduction_data['actual_loads']),
                'total_reduction': sum(reduction_data['hourly_reductions']),
                'performance_ratio': sum(reduction_data['hourly_reductions']) / 
                                   (reduction_data['committed_reduction'] * len(reduction_data['event_hours']))
                                   if reduction_data['committed_reduction'] > 0 else 0
            },
            'payment_breakdown': payments,
            'payment_status': 'APPROVED' if payments['total'] > 0 else 'REVIEW'
        }
        return invoice

# Generate sample data
print("\n1. Setting Up Building Profile...")
print("-" * 40)

np.random.seed(42)

# Create 30 days of historical load data
end_date = datetime(2024, 9, 15)
start_date = end_date - timedelta(days=30)
timestamps = pd.date_range(start_date, end_date, freq='h')

# Generate commercial building load profile
building_loads = []
for ts in timestamps:
    hour = ts.hour
    day_of_week = ts.dayofweek
    
    if day_of_week < 5:  # Weekday
        if 8 <= hour <= 18:
            base_load = 40
        else:
            base_load = 25
    else:  # Weekend
        base_load = 20
    
    # Add variation
    load = base_load + np.random.normal(0, 3)
    building_loads.append(max(15, load))

historical_load = pd.Series(building_loads, index=timestamps, name='load_kw')

customer_info = {
    'name': 'Example Commercial Building',
    'account': 'COM-12345',
    'enrolled_capacity': 10,  # 10 kW enrolled
    'dr_program': 'Critical Peak Pricing'
}

print(f"Customer: {customer_info['name']}")
print(f"Account: {customer_info['account']}")
print(f"Enrolled Capacity: {customer_info['enrolled_capacity']} kW")
print(f"Average Load: {historical_load.mean():.1f} kW")

# Simulate DR event
event_date = datetime(2024, 9, 15)
event_hours = [14, 15, 16, 17]  # 2 PM - 6 PM

print(f"\n2. Demand Response Event Simulation...")
print("-" * 40)
print(f"Event Date: {event_date.date()}")
print(f"Event Hours: {event_hours[0]}:00 - {event_hours[-1]+1}:00")

# Calculate baseline
calculator = PaymentCalculator()
baseline = calculator.calculate_baseline(historical_load, event_date, event_hours)
print(f"Calculated Baseline: {baseline:.1f} kW")

# Simulate actual loads during event (with reduction)
actual_loads = []
for hour in range(24):
    if hour in event_hours:
        # Reduce by 25% during event
        reduction_factor = 0.75
        load = baseline * reduction_factor + np.random.normal(0, 2)
    else:
        # Normal operation
        if 8 <= hour <= 18:
            load = 40 + np.random.normal(0, 3)
        else:
            load = 25 + np.random.normal(0, 3)
    actual_loads.append(max(10, load))

# Calculate reductions
hourly_reductions = calculator.calculate_reduction(baseline, actual_loads, event_hours)

reduction_data = {
    'event_date': event_date.isoformat(),
    'event_hours': event_hours,
    'baseline': baseline,
    'actual_loads': [actual_loads[h] for h in event_hours],
    'hourly_reductions': hourly_reductions,
    'enrolled_capacity': customer_info['enrolled_capacity'],
    'committed_reduction': 8  # 8 kW committed
}

print(f"Actual Load During Event: {np.mean([actual_loads[h] for h in event_hours]):.1f} kW")
print(f"Total Reduction Achieved: {sum(hourly_reductions):.1f} kWh")
print(f"Average Reduction: {np.mean(hourly_reductions):.1f} kW")

# Calculate payments
print("\n3. Payment Calculation...")
print("-" * 40)

payments = calculator.calculate_payment(reduction_data)

for payment_type, amount in payments.items():
    if payment_type != 'total' and amount != 0:
        print(f"{payment_type.replace('_', ' ').title():20} ${amount:8.2f}")

print("-" * 40)
print(f"{'Total Payment':20} ${payments['total']:8.2f}")

# Generate invoice
invoice = calculator.generate_invoice(customer_info, reduction_data, payments)

# Save invoice
with open('SCAN/demand_response/payment_calc/invoice.json', 'w') as f:
    json.dump(invoice, f, indent=2, default=str)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Load profile during event day
ax = axes[0, 0]
hours = range(24)
baseline_line = [baseline if h in event_hours else np.nan for h in hours]
ax.plot(hours, actual_loads, label='Actual Load', linewidth=2, marker='o', markersize=4)
ax.plot(hours, baseline_line, label='Baseline', linewidth=2, linestyle='--', color='red')

# Fill reduction area
for i, hour in enumerate(event_hours):
    ax.fill_between([hour, hour+1], 
                    [actual_loads[hour], actual_loads[hour]], 
                    [baseline, baseline], 
                    alpha=0.3, color='green')

ax.set_xlabel('Hour of Day')
ax.set_ylabel('Load (kW)')
ax.set_title(f'DR Event Performance - {event_date.date()}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Hourly reductions
ax = axes[0, 1]
event_hour_labels = [f"{h}:00" for h in event_hours]
colors = ['green' if r > 0 else 'red' for r in hourly_reductions]
ax.bar(event_hour_labels, hourly_reductions, color=colors, alpha=0.7)
ax.axhline(y=reduction_data['committed_reduction'], color='blue', linestyle='--', 
          label=f"Committed: {reduction_data['committed_reduction']} kW")
ax.set_xlabel('Hour')
ax.set_ylabel('Reduction (kW)')
ax.set_title('Hourly Load Reductions')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Payment breakdown
ax = axes[0, 2]
payment_types = [k for k in payments.keys() if k != 'total' and payments[k] != 0]
payment_amounts = [payments[k] for k in payment_types]
colors_pay = ['green' if a > 0 else 'red' for a in payment_amounts]
ax.barh(payment_types, np.abs(payment_amounts), color=colors_pay, alpha=0.7)
ax.set_xlabel('Payment ($)')
ax.set_title('Payment Breakdown')
ax.grid(True, alpha=0.3, axis='x')

# Plot 4: Historical baseline calculation
ax = axes[1, 0]
baseline_days = []
for i in range(1, 11):
    day = event_date - timedelta(days=i)
    if day.weekday() < 5:
        baseline_days.append(day)
        if len(baseline_days) <= 5:
            day_loads = [historical_load.loc[day.replace(hour=h)] 
                        if day.replace(hour=h) in historical_load.index else np.nan
                        for h in event_hours]
            ax.plot(event_hours, day_loads, alpha=0.3, color='gray')

ax.axhline(y=baseline, color='red', linewidth=2, label='Calculated Baseline')
ax.set_xlabel('Hour')
ax.set_ylabel('Load (kW)')
ax.set_title('Historical Days for Baseline')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Performance metrics
ax = axes[1, 1]
metrics = {
    'Enrolled\nCapacity': customer_info['enrolled_capacity'],
    'Committed\nReduction': reduction_data['committed_reduction'],
    'Achieved\nReduction': np.mean(hourly_reductions),
    'Performance\nRatio': (sum(hourly_reductions) / 
                          (reduction_data['committed_reduction'] * len(event_hours))) * 100
}

bars = ax.bar(range(len(metrics)), list(metrics.values()), color='steelblue', alpha=0.7)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(list(metrics.keys()))
ax.set_ylabel('Value')
ax.set_title('Performance Metrics')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, metrics.values())):
    if i == 3:  # Performance ratio
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{value:.1f}%', ha='center', va='bottom')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f'{value:.1f}', ha='center', va='bottom')

ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Monthly earnings projection
ax = axes[1, 2]
events_per_month = 4  # Assume 4 events per month
monthly_projections = {
    'Capacity': payments['capacity'],
    'Energy': payments['energy'] * events_per_month,
    'Performance': payments['performance'] * events_per_month,
    'Peak Bonus': payments['peak'] * events_per_month,
    'Total': payments['capacity'] + (payments['energy'] + payments['performance'] + 
                                      payments['peak']) * events_per_month
}

ax.bar(range(len(monthly_projections)), list(monthly_projections.values()), 
       color=['blue', 'green', 'orange', 'red', 'purple'], alpha=0.7)
ax.set_xticks(range(len(monthly_projections)))
ax.set_xticklabels(list(monthly_projections.keys()), rotation=45)
ax.set_ylabel('Projected Earnings ($)')
ax.set_title(f'Monthly Earnings Projection ({events_per_month} events/month)')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, value in enumerate(monthly_projections.values()):
    ax.text(i, value + 5, f'${value:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('SCAN/demand_response/payment_calc/payment_analysis.png', dpi=150, bbox_inches='tight')

# Create summary report
summary = {
    'event_summary': {
        'date': event_date.isoformat(),
        'duration_hours': len(event_hours),
        'baseline_kw': round(baseline, 2),
        'average_actual_kw': round(np.mean([actual_loads[h] for h in event_hours]), 2),
        'total_reduction_kwh': round(sum(hourly_reductions), 2),
        'average_reduction_kw': round(np.mean(hourly_reductions), 2),
        'reduction_percentage': round((np.mean(hourly_reductions) / baseline) * 100, 1)
    },
    'performance_metrics': {
        'committed_kw': reduction_data['committed_reduction'],
        'achieved_kw': round(np.mean(hourly_reductions), 2),
        'performance_ratio': round((sum(hourly_reductions) / 
                                   (reduction_data['committed_reduction'] * len(event_hours))), 3),
        'compliance': 'PASSED' if np.mean(hourly_reductions) >= reduction_data['committed_reduction'] * 0.9 
                     else 'FAILED'
    },
    'payment_summary': {
        'event_payment': round(payments['energy'] + payments['performance'] + payments['peak'], 2),
        'monthly_capacity': round(payments['capacity'], 2),
        'total_payment': round(payments['total'], 2),
        'annual_projection': round(payments['capacity'] * 12 + 
                                  (payments['energy'] + payments['performance'] + payments['peak']) * 48, 2)
    }
}

# Save summary
with open('SCAN/demand_response/payment_calc/payment_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n4. Annual Projection...")
print("-" * 40)
print(f"Events per year: 48 (estimated)")
print(f"Annual capacity payment: ${payments['capacity'] * 12:,.2f}")
print(f"Annual energy payments: ${(payments['energy'] + payments['performance'] + payments['peak']) * 48:,.2f}")
print(f"Total annual earnings: ${summary['payment_summary']['annual_projection']:,.2f}")

print("\n" + "=" * 60)
print("Payment calculation complete!")
print("Results saved to:")
print("  - SCAN/demand_response/payment_calc/payment_analysis.png")
print("  - SCAN/demand_response/payment_calc/invoice.json")
print("  - SCAN/demand_response/payment_calc/payment_summary.json")
print("=" * 60)