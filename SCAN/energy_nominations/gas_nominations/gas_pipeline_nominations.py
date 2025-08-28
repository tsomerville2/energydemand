"""
Gas Pipeline Nomination System
Real implementation of NAESB-compliant gas nominations
Demonstrates nomination creation, validation, and scheduling
"""

import pandas as pd
import json
from datetime import datetime, timedelta, time
import uuid
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("GAS PIPELINE NOMINATION SYSTEM")
print("=" * 60)

class GasNomination:
    """NAESB-compliant gas nomination"""
    
    def __init__(self):
        # NAESB standard gas day: 9 AM - 9 AM Central Time
        self.gas_day_start = time(9, 0)  # 9 AM CT
        self.nomination_cycles = {
            'TIMELY': {'deadline': time(11, 30), 'effective': time(9, 0)},  # 11:30 AM CT day before
            'EVENING': {'deadline': time(18, 0), 'effective': time(9, 0)},  # 6:00 PM CT day before
            'INTRADAY1': {'deadline': time(10, 0), 'effective': time(14, 0)},  # 10:00 AM CT same day
            'INTRADAY2': {'deadline': time(17, 30), 'effective': time(21, 0)},  # 5:30 PM CT same day
            'INTRADAY3': {'deadline': time(7, 0), 'effective': time(9, 0)}  # 7:00 AM CT next day
        }
        
    def create_nomination(self, shipper_name, pipeline, gas_day, cycle='TIMELY'):
        """Create a NAESB-compliant nomination"""
        
        nomination = {
            'header': {
                'transaction_id': f"NOM-{uuid.uuid4().hex[:8].upper()}",
                'transaction_type': 'NOMINATION',
                'version': 'NAESB-5.0',
                'timestamp': datetime.now().isoformat(),
                'gas_day': gas_day.isoformat(),
                'cycle': cycle
            },
            'shipper': {
                'name': shipper_name,
                'contract_number': 'GC-2024-001',
                'duns_number': '123456789'  # Sample DUNS
            },
            'pipeline': {
                'name': pipeline,
                'tsp_code': self._get_tsp_code(pipeline),
                'proprietary_code': 'PIPE-001'
            },
            'locations': [],
            'package': {
                'total_receipts': 0,
                'total_deliveries': 0,
                'imbalance': 0
            }
        }
        
        return nomination
    
    def _get_tsp_code(self, pipeline_name):
        """Get Transportation Service Provider code"""
        tsp_codes = {
            'TransCanada': 'TCPL',
            'Kinder Morgan': 'KM',
            'Enterprise': 'EPD',
            'Williams': 'WMB',
            'Enbridge': 'ENB'
        }
        return tsp_codes.get(pipeline_name, 'UNKN')
    
    def add_receipt_point(self, nomination, location, quantity, rank=1):
        """Add a receipt point to nomination"""
        receipt = {
            'location_type': 'RECEIPT',
            'location_name': location,
            'location_code': f"R-{location[:4].upper()}",
            'quantity_dth': quantity,  # Dekatherms
            'rank': rank,
            'upstream_party': 'PRODUCER_001',
            'upstream_contract': 'PC-2024-001',
            'measurement_basis': 'DRY',
            'fuel_rate': 0.02  # 2% fuel retention
        }
        nomination['locations'].append(receipt)
        nomination['package']['total_receipts'] += quantity
        self._update_imbalance(nomination)
        return nomination
    
    def add_delivery_point(self, nomination, location, quantity, rank=1):
        """Add a delivery point to nomination"""
        delivery = {
            'location_type': 'DELIVERY',
            'location_name': location,
            'location_code': f"D-{location[:4].upper()}",
            'quantity_dth': quantity,
            'rank': rank,
            'downstream_party': 'LDC_001',  # Local Distribution Company
            'downstream_contract': 'DC-2024-001',
            'delivery_pressure_psi': 500,
            'max_hourly_quantity': quantity / 24
        }
        nomination['locations'].append(delivery)
        nomination['package']['total_deliveries'] += quantity
        self._update_imbalance(nomination)
        return nomination
    
    def _update_imbalance(self, nomination):
        """Calculate nomination imbalance"""
        nomination['package']['imbalance'] = (
            nomination['package']['total_receipts'] - 
            nomination['package']['total_deliveries']
        )
    
    def validate_nomination(self, nomination):
        """Validate nomination against NAESB business rules"""
        errors = []
        warnings = []
        
        # Check balance
        imbalance = nomination['package']['imbalance']
        if abs(imbalance) > 0.01:  # Allow 0.01 DTH tolerance
            errors.append(f"Nomination imbalanced by {imbalance:.2f} DTH")
        
        # Check minimum quantities
        if nomination['package']['total_receipts'] < 100:
            warnings.append("Total receipts below typical minimum (100 DTH)")
        
        # Check rank sequences
        receipt_ranks = [loc['rank'] for loc in nomination['locations'] if loc['location_type'] == 'RECEIPT']
        if receipt_ranks != sorted(receipt_ranks):
            warnings.append("Receipt ranks not in sequential order")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

class PipelineScheduler:
    """Pipeline scheduling system"""
    
    def __init__(self):
        self.scheduled_capacity = {}
        self.available_capacity = {
            'TransCanada': 5000000,  # 5 BCF/day
            'Kinder Morgan': 3000000,
            'Enterprise': 2000000,
            'Williams': 4000000,
            'Enbridge': 3500000
        }
    
    def process_nominations(self, nominations):
        """Process and confirm nominations"""
        confirmations = []
        
        for nom in nominations:
            pipeline = nom['pipeline']['name']
            requested = nom['package']['total_receipts']
            
            # Check available capacity
            used = self.scheduled_capacity.get(pipeline, 0)
            available = self.available_capacity.get(pipeline, 0) - used
            
            if requested <= available:
                scheduled = requested
                status = 'CONFIRMED'
            else:
                scheduled = available
                status = 'PARTIAL'
            
            confirmation = {
                'transaction_id': nom['header']['transaction_id'],
                'confirmation_id': f"CONF-{uuid.uuid4().hex[:8].upper()}",
                'status': status,
                'requested_quantity': requested,
                'scheduled_quantity': scheduled,
                'effective_time': nom['header']['gas_day'],
                'pipeline': pipeline
            }
            
            confirmations.append(confirmation)
            self.scheduled_capacity[pipeline] = used + scheduled
        
        return confirmations

# Create nomination system
nom_system = GasNomination()
scheduler = PipelineScheduler()

# Example 1: Create a balanced nomination
print("\n1. Creating Balanced Gas Nomination:")
print("-" * 40)

gas_day = datetime.now().date() + timedelta(days=1)
nomination1 = nom_system.create_nomination(
    shipper_name="EXAMPLE_GAS_MARKETING",
    pipeline="TransCanada",
    gas_day=gas_day,
    cycle='TIMELY'
)

# Add receipt points (where gas enters the pipeline)
nomination1 = nom_system.add_receipt_point(nomination1, "Permian Hub", 50000, rank=1)
nomination1 = nom_system.add_receipt_point(nomination1, "Marcellus Pool", 30000, rank=2)

# Add delivery points (where gas exits the pipeline)
nomination1 = nom_system.add_delivery_point(nomination1, "Chicago Citygate", 45000, rank=1)
nomination1 = nom_system.add_delivery_point(nomination1, "Detroit Metro", 35000, rank=2)

# Validate
validation1 = nom_system.validate_nomination(nomination1)

print(f"Transaction ID: {nomination1['header']['transaction_id']}")
print(f"Pipeline: {nomination1['pipeline']['name']}")
print(f"Gas Day: {nomination1['header']['gas_day']}")
print(f"Total Receipts: {nomination1['package']['total_receipts']:,} DTH")
print(f"Total Deliveries: {nomination1['package']['total_deliveries']:,} DTH")
print(f"Imbalance: {nomination1['package']['imbalance']:,} DTH")
print(f"Validation: {'✅ VALID' if validation1['valid'] else '❌ INVALID'}")
if validation1['errors']:
    for error in validation1['errors']:
        print(f"  ERROR: {error}")
if validation1['warnings']:
    for warning in validation1['warnings']:
        print(f"  WARNING: {warning}")

# Example 2: Create multiple nominations for scheduling
print("\n2. Creating Multiple Nominations for Different Pipelines:")
print("-" * 40)

all_nominations = [nomination1]

# Create additional nominations
pipelines = ['Kinder Morgan', 'Williams', 'Enterprise']
quantities = [(25000, 25000), (40000, 40000), (15000, 15000)]

for pipeline, (receipt_qty, delivery_qty) in zip(pipelines, quantities):
    nom = nom_system.create_nomination(
        shipper_name="EXAMPLE_GAS_MARKETING",
        pipeline=pipeline,
        gas_day=gas_day,
        cycle='TIMELY'
    )
    nom = nom_system.add_receipt_point(nom, f"{pipeline} Receipt", receipt_qty)
    nom = nom_system.add_delivery_point(nom, f"{pipeline} Delivery", delivery_qty)
    all_nominations.append(nom)
    
    print(f"  {pipeline}: {receipt_qty:,} DTH nominated")

# Process all nominations through scheduler
print("\n3. Processing Nominations through Pipeline Scheduler:")
print("-" * 40)

confirmations = scheduler.process_nominations(all_nominations)

for conf in confirmations:
    print(f"Pipeline: {conf['pipeline']:15} | Status: {conf['status']:10} | "
          f"Requested: {conf['requested_quantity']:7,} DTH | "
          f"Scheduled: {conf['scheduled_quantity']:7,} DTH")

# Save nominations and confirmations
with open('SCAN/energy_nominations/gas_nominations/sample_nomination.json', 'w') as f:
    json.dump(nomination1, f, indent=2, default=str)

with open('SCAN/energy_nominations/gas_nominations/confirmations.json', 'w') as f:
    json.dump(confirmations, f, indent=2, default=str)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Nomination by Pipeline
ax = axes[0, 0]
pipelines = [conf['pipeline'] for conf in confirmations]
requested = [conf['requested_quantity'] for conf in confirmations]
scheduled = [conf['scheduled_quantity'] for conf in confirmations]

x = range(len(pipelines))
width = 0.35
ax.bar([i - width/2 for i in x], requested, width, label='Requested', color='steelblue', alpha=0.8)
ax.bar([i + width/2 for i in x], scheduled, width, label='Scheduled', color='green', alpha=0.8)
ax.set_xlabel('Pipeline')
ax.set_ylabel('Quantity (DTH)')
ax.set_title('Gas Nominations by Pipeline')
ax.set_xticks(x)
ax.set_xticklabels(pipelines, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Capacity Utilization
ax = axes[0, 1]
utilization = [(scheduler.scheduled_capacity.get(p, 0) / scheduler.available_capacity[p]) * 100 
               for p in scheduler.available_capacity.keys()]
pipeline_names = list(scheduler.available_capacity.keys())

colors = ['green' if u < 80 else 'yellow' if u < 95 else 'red' for u in utilization]
ax.barh(pipeline_names, utilization, color=colors, alpha=0.7)
ax.set_xlabel('Capacity Utilization (%)')
ax.set_title('Pipeline Capacity Utilization')
ax.grid(True, alpha=0.3, axis='x')
for i, (p, u) in enumerate(zip(pipeline_names, utilization)):
    ax.text(u + 1, i, f'{u:.1f}%', va='center')

# Plot 3: Nomination Flow Diagram for nomination1
ax = axes[1, 0]
receipt_locs = [loc['location_name'] for loc in nomination1['locations'] if loc['location_type'] == 'RECEIPT']
receipt_qtys = [loc['quantity_dth'] for loc in nomination1['locations'] if loc['location_type'] == 'RECEIPT']
delivery_locs = [loc['location_name'] for loc in nomination1['locations'] if loc['location_type'] == 'DELIVERY']
delivery_qtys = [loc['quantity_dth'] for loc in nomination1['locations'] if loc['location_type'] == 'DELIVERY']

# Create flow visualization
y_receipts = range(len(receipt_locs))
y_deliveries = range(len(delivery_locs))

# Plot receipt points
for i, (loc, qty) in enumerate(zip(receipt_locs, receipt_qtys)):
    ax.barh(i, -qty/1000, height=0.3, color='blue', alpha=0.7)
    ax.text(-qty/1000 - 2, i, f"{loc}\n{qty:,} DTH", ha='right', va='center', fontsize=8)

# Plot delivery points
for i, (loc, qty) in enumerate(zip(delivery_locs, delivery_qtys)):
    ax.barh(i - 0.5, qty/1000, height=0.3, color='green', alpha=0.7)
    ax.text(qty/1000 + 2, i - 0.5, f"{loc}\n{qty:,} DTH", ha='left', va='center', fontsize=8)

ax.axvline(x=0, color='black', linewidth=2)
ax.text(0, len(receipt_locs), nomination1['pipeline']['name'], ha='center', fontweight='bold')
ax.set_xlabel('Flow (thousands DTH)')
ax.set_title('Nomination Flow Diagram')
ax.set_yticks([])
ax.grid(True, alpha=0.3, axis='x')

# Plot 4: NAESB Nomination Timeline
ax = axes[1, 1]
cycles = list(nom_system.nomination_cycles.keys())
deadlines = [nom_system.nomination_cycles[c]['deadline'].hour + nom_system.nomination_cycles[c]['deadline'].minute/60 
             for c in cycles]
effective = [nom_system.nomination_cycles[c]['effective'].hour + nom_system.nomination_cycles[c]['effective'].minute/60
             for c in cycles]

ax.scatter(deadlines, range(len(cycles)), s=100, c='red', marker='v', label='Deadline', zorder=3)
ax.scatter(effective, range(len(cycles)), s=100, c='green', marker='^', label='Effective', zorder=3)

for i, cycle in enumerate(cycles):
    ax.plot([deadlines[i], effective[i]], [i, i], 'k--', alpha=0.3)
    ax.text(deadlines[i], i + 0.15, f"{nom_system.nomination_cycles[cycle]['deadline'].strftime('%H:%M')}", 
           ha='center', fontsize=8)
    ax.text(effective[i], i - 0.15, f"{nom_system.nomination_cycles[cycle]['effective'].strftime('%H:%M')}", 
           ha='center', fontsize=8)

ax.set_yticks(range(len(cycles)))
ax.set_yticklabels(cycles)
ax.set_xlabel('Hour of Day (24h format)')
ax.set_title('NAESB Nomination Cycles Timeline')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 25)

plt.tight_layout()
plt.savefig('SCAN/energy_nominations/gas_nominations/gas_nomination_system.png', dpi=150, bbox_inches='tight')

# Create summary report
summary = pd.DataFrame(confirmations)
summary.to_csv('SCAN/energy_nominations/gas_nominations/nomination_summary.csv', index=False)

print("\n" + "=" * 60)
print("Gas Nomination System Complete!")
print("Results saved to:")
print("  - SCAN/energy_nominations/gas_nominations/sample_nomination.json")
print("  - SCAN/energy_nominations/gas_nominations/confirmations.json")
print("  - SCAN/energy_nominations/gas_nominations/gas_nomination_system.png")
print("  - SCAN/energy_nominations/gas_nominations/nomination_summary.csv")
print("=" * 60)