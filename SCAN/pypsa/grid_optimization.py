"""
PyPSA Grid Optimization Demo
Real implementation with actual data for a 3-bus power system
Demonstrates optimal power flow with renewable generation
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create a new network
network = pypsa.Network()

# Set snapshots (24 hours of hourly data)
hours = pd.date_range('2024-01-01', periods=24, freq='h')
network.set_snapshots(hours)

# Add three buses (nodes in the network)
buses = ['Bus_North', 'Bus_Central', 'Bus_South']
for bus in buses:
    network.add("Bus", bus, v_nom=230)  # 230 kV voltage level

# Add transmission lines between buses
network.add("Line", "Line_N-C",
           bus0="Bus_North", bus1="Bus_Central",
           x=0.1, r=0.01,  # Reactance and resistance
           s_nom=500)  # 500 MW capacity

network.add("Line", "Line_C-S", 
           bus0="Bus_Central", bus1="Bus_South",
           x=0.1, r=0.01,
           s_nom=400)

network.add("Line", "Line_N-S",
           bus0="Bus_North", bus1="Bus_South",
           x=0.15, r=0.015,
           s_nom=300)

# Add generators with different technologies
# Solar at North Bus
solar_profile = np.array([0, 0, 0, 0, 0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0,
                         0.95, 0.85, 0.7, 0.5, 0.3, 0.1, 0, 0, 0, 0, 0, 0])
network.add("Generator", "Solar_Farm",
           bus="Bus_North",
           p_nom=200,  # 200 MW capacity
           marginal_cost=0,  # Solar has zero marginal cost
           p_max_pu=solar_profile)

# Wind at South Bus
# More realistic wind profile with variability
np.random.seed(42)
wind_profile = 0.3 + 0.4 * np.sin(np.linspace(0, 2*np.pi, 24)) + 0.2 * np.random.random(24)
wind_profile = np.clip(wind_profile, 0, 1)
network.add("Generator", "Wind_Farm",
           bus="Bus_South", 
           p_nom=250,  # 250 MW capacity
           marginal_cost=0,  # Wind has zero marginal cost
           p_max_pu=wind_profile)

# Gas generator at Central Bus (dispatchable)
network.add("Generator", "Gas_Plant",
           bus="Bus_Central",
           p_nom=400,  # 400 MW capacity
           marginal_cost=50)  # $50/MWh

# Add coal backup at Central Bus (more expensive)
network.add("Generator", "Coal_Plant",
           bus="Bus_Central",
           p_nom=300,
           marginal_cost=80)  # $80/MWh

# Add loads (demand) at each bus
# Typical daily load profile
base_load = np.array([300, 280, 270, 260, 270, 300, 350, 400, 420, 430, 425, 420,
                     410, 405, 400, 410, 430, 450, 440, 420, 380, 350, 330, 310])

network.add("Load", "Load_North",
           bus="Bus_North",
           p_set=base_load * 0.3)  # 30% of total load

network.add("Load", "Load_Central",
           bus="Bus_Central", 
           p_set=base_load * 0.5)  # 50% of total load

network.add("Load", "Load_South",
           bus="Bus_South",
           p_set=base_load * 0.2)  # 20% of total load

# Solve optimal power flow
network.optimize(solver_name='highs')

# Print results
print("=" * 60)
print("PYPSA GRID OPTIMIZATION RESULTS")
print("=" * 60)
print(f"\nTotal system cost: ${network.objective:.2f}")
print(f"Average electricity price: ${network.objective / network.loads_t.p_set.sum().sum():.2f}/MWh")

# Generator dispatch summary
print("\nGenerator Dispatch Summary:")
print("-" * 40)
for gen in network.generators.index:
    total_generation = network.generators_t.p[gen].sum()
    capacity_factor = total_generation / (network.generators.at[gen, 'p_nom'] * 24) * 100
    cost = network.generators.at[gen, 'marginal_cost'] * total_generation
    print(f"{gen:15} | Generated: {total_generation:8.1f} MWh | CF: {capacity_factor:5.1f}% | Cost: ${cost:,.0f}")

# Line utilization
print("\nTransmission Line Utilization:")
print("-" * 40)
for line in network.lines.index:
    max_flow = network.lines_t.p0[line].abs().max()
    avg_utilization = (network.lines_t.p0[line].abs().mean() / network.lines.at[line, 's_nom']) * 100
    print(f"{line:10} | Max Flow: {max_flow:6.1f} MW | Avg Utilization: {avg_utilization:5.1f}%")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Generation by source
ax = axes[0, 0]
for gen in network.generators.index:
    ax.plot(hours, network.generators_t.p[gen], label=gen, linewidth=2)
ax.set_xlabel('Hour')
ax.set_ylabel('Generation (MW)')
ax.set_title('Hourly Generation by Source')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Load vs Total Generation
ax = axes[0, 1]
total_load = network.loads_t.p_set.sum(axis=1)
total_gen = network.generators_t.p.sum(axis=1)
ax.plot(hours, total_load, label='Total Load', linewidth=2, color='red')
ax.plot(hours, total_gen, label='Total Generation', linewidth=2, color='green', linestyle='--')
ax.set_xlabel('Hour')
ax.set_ylabel('Power (MW)')
ax.set_title('Load vs Generation Balance')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Renewable penetration
ax = axes[1, 0]
renewable_gen = network.generators_t.p[['Solar_Farm', 'Wind_Farm']].sum(axis=1)
renewable_percentage = (renewable_gen / total_gen) * 100
ax.fill_between(hours, renewable_percentage, alpha=0.5, color='green')
ax.plot(hours, renewable_percentage, linewidth=2, color='darkgreen')
ax.set_xlabel('Hour')
ax.set_ylabel('Renewable Share (%)')
ax.set_title('Renewable Energy Penetration')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

# Plot 4: Marginal prices at each bus
ax = axes[1, 1]
for bus in network.buses.index:
    marginal_price = network.buses_t.marginal_price[bus]
    ax.plot(hours, marginal_price, label=bus, linewidth=2)
ax.set_xlabel('Hour')
ax.set_ylabel('Marginal Price ($/MWh)')
ax.set_title('Locational Marginal Prices')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SCAN/pypsa/grid_optimization_results.png', dpi=150, bbox_inches='tight')
# plt.show()  # Commented to prevent blocking

# Save detailed results to CSV
results_df = pd.DataFrame({
    'Hour': hours,
    'Solar_MW': network.generators_t.p['Solar_Farm'],
    'Wind_MW': network.generators_t.p['Wind_Farm'],
    'Gas_MW': network.generators_t.p['Gas_Plant'],
    'Coal_MW': network.generators_t.p['Coal_Plant'],
    'Total_Load_MW': total_load,
    'Renewable_Percentage': renewable_percentage,
    'Avg_Marginal_Price': network.buses_t.marginal_price.mean(axis=1)
})
results_df.to_csv('SCAN/pypsa/hourly_results.csv', index=False)

print("\n" + "=" * 60)
print("Results saved to:")
print("  - SCAN/pypsa/grid_optimization_results.png")
print("  - SCAN/pypsa/hourly_results.csv")
print("=" * 60)