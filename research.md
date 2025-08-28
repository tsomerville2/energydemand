# Open Source Projects for Energy Markets, Interval Data, and Baseline Estimation

## Executive Summary
This document catalogs open source projects suitable for energy market analysis, interval data processing, baseline estimation, and event prediction. All projects listed use permissive licenses (MIT, Apache 2.0, MPL 2.0) that allow commercial use without requiring disclosure of proprietary data or code modifications.

## License Overview for Commercial Use

### Key License Permissions for Your Use Case

**MIT License** (Most Permissive)
- ✅ Use with proprietary data without sharing results
- ✅ Modify code without sharing modifications  
- ✅ Incorporate into proprietary commercial products
- ✅ Sublicense and sell derivative works
- ⚠️ Must include original copyright notice and license text in your distribution
- ⚠️ No warranty provided

**Apache 2.0 License** (Patent Protection)
- ✅ Use with proprietary data without sharing results
- ✅ Modify code without sharing modifications
- ✅ Patent grant protects against patent litigation
- ✅ Incorporate into proprietary products
- ⚠️ Must include original copyright, license text, and NOTICE file if present
- ⚠️ Must state significant changes made to the code
- ⚠️ No warranty provided

**Mozilla Public License 2.0** (File-Level Copyleft)
- ✅ Use with proprietary data without sharing results
- ✅ Can keep modifications private IF you don't distribute the modified MPL files
- ✅ Can combine with proprietary code (larger work remains proprietary)
- ⚠️ If you distribute modified MPL-licensed files, must share those file modifications
- ⚠️ Can work around by keeping MPL code as separate library/service
- ⚠️ No warranty provided

---

## 1. Production-Ready Forecasting Systems

### OpenSTEF - Open Short Term Energy Forecasting
- **License**: Mozilla Public License 2.0
- **Repository**: `https://github.com/OpenSTEF/openstef`
- **Commercial Use**: Can use internally without sharing modifications. If distributing, only modified MPL files need to be shared.
- **Key Features**:
  - Automated ML pipelines for 48-hour ahead grid load forecasting
  - Handles interval data at hourly and sub-hourly resolution
  - Integrates weather data, market prices, and historical measurements
  - Probabilistic forecasts with confidence intervals
  - Microservice architecture optimized for cloud deployment
  - REST API and expert GUI included
  - Production-tested at Alliander (Dutch DSO) for congestion management
- **Data Handling**: 
  - Input: Time series of measured load/generation
  - Output: 48-hour probabilistic forecasts
  - Supports both demand and distributed generation

### PyPSA - Python for Power System Analysis
- **License**: MIT License
- **Repository**: `https://github.com/PyPSA/PyPSA`
- **Commercial Use**: Full freedom to modify and use without sharing anything
- **Key Features**:
  - Optimizes power system operation and investment
  - Handles unit commitment, optimal power flow, security-constrained optimization
  - Models renewable variability, storage, sector coupling
  - Scales to continental-size networks
  - Integrates with renewable.ninja for weather data
- **Data Handling**:
  - Supports multi-year hourly time series
  - Handles networks with 10,000+ buses
  - Includes European grid topology data

---

## 2. Baseline Estimation & Energy Measurement

### CalTRACK / OpenEEmeter
- **License**: Apache 2.0
- **Repository**: `https://github.com/openeemeter/eemeter`
- **Commercial Use**: Full freedom to use and modify. Must acknowledge changes if distributing.
- **Key Features**:
  - Industry-standard baseline calculation methods
  - Weather normalization using TMY3 data
  - Supports both monthly billing and interval (15-min, hourly, daily) data
  - Calculates avoided energy use for efficiency programs
  - Statistical confidence bounds on savings estimates
  - Month-by-month baseline models for seasonal accuracy
- **Implementation Details**:
  - Based on PRISM (1986) for daily/billing models
  - TOWT model (2011) for hourly models
  - Tested on 4,777 residential retrofits
  - Meets IPMVP Option C standards

### NILMTK - Non-Intrusive Load Monitoring Toolkit
- **License**: Apache 2.0 (implied from documentation)
- **Repository**: `https://github.com/nilmtk/nilmtk`
- **Commercial Use**: Can modify and use privately without disclosure
- **Key Features**:
  - Energy disaggregation from aggregate meter data
  - Identifies individual appliance consumption patterns
  - Benchmarking framework for algorithm comparison
  - Data converters for 8+ public datasets
  - Metrics: accuracy, F1-score, estimation accuracy
- **Supported Datasets**:
  - UK-DALE: 5 UK homes, 2-year data, 16kHz & 1/6 Hz sampling
  - REDD: 6 US homes, up to 119 days, 15 kHz & 1 Hz
  - REFIT: 20 UK homes, 2 years, 8-second intervals
  - AMPds, GREEND, WikiEnergy, iAWE datasets

---

## 3. HuggingFace Models & Datasets

### LSTEnergy Model
- **License**: Check individual model cards
- **Repository**: `https://huggingface.co/databloom/LSTEnergy`
- **Key Features**:
  - LSTM architecture for energy consumption forecasting
  - Trained on 12M+ smartmeter instances
  - Customizable prediction windows
  - Includes baseline comparison models
- **Data Format**: 
  - Input: id, device_name, property, value, timestamp
  - Output: Consumption forecasts with confidence intervals

### Electricity Load Diagrams Dataset
- **License**: Check dataset card
- **Repository**: `https://huggingface.co/datasets/electricity_load_diagrams`
- **Details**:
  - 370 Portuguese clients, 2011-2014
  - Hourly kW consumption
  - 140,256 time points
  - Includes train/validation/test splits

### ETT (Electricity Transformer Temperature)
- **License**: Check dataset card
- **Repository**: `https://huggingface.co/datasets/ett`
- **Details**:
  - 2-year data from electricity transformers
  - 7 features: oil temperature, load, weather
  - 15-minute and hourly resolutions
  - Designed for long-term forecasting research

---

## 4. Building Energy Benchmarking

### BETTER - Building Efficiency Targeting Tool
- **License**: Modified BSD (check repository for details)
- **Repository**: `https://github.com/LBNL-JCI-ICF/better`
- **Commercial Use**: Generally permissive like standard BSD
- **Key Features**:
  - Automated virtual energy audits at portfolio scale
  - Identifies retrofit opportunities without site visits
  - Statistical baseline from peer buildings
  - Python 3.6+ implementation
  - API for integration with building management systems
- **Capabilities**:
  - Processes monthly utility bills or interval data
  - Weather normalization across climate zones
  - Generates Energy Star scores
  - Prioritizes ECMs (Energy Conservation Measures)

### Energym
- **License**: MIT License
- **Repository**: `https://github.com/bsl546/energym`
- **Commercial Use**: Full freedom to modify and keep private
- **Key Features**:
  - Building simulation library for control strategy testing
  - Benchmark suite for HVAC controllers
  - Gym-like interface for reinforcement learning
  - Calibrated models from real buildings
  - Weather data integration
- **Building Models**:
  - Office buildings (small, medium, large)
  - Mixed-use buildings
  - Data centers
  - Schools and hospitals

### OpenEMS - Open Energy Management System
- **License**: Eclipse Public License 2.0
- **Repository**: `https://github.com/OpenEMS/openems`
- **Commercial Use**: Can modify and use commercially. EPL is business-friendly.
- **Key Features**:
  - Modular platform for energy management applications
  - Real-time data collection from meters, inverters, batteries
  - Edge computing architecture
  - MODBUS, REST API support
  - Time-series database integration
- **Applications**:
  - Peak shaving
  - Self-consumption optimization
  - Grid services (frequency regulation)
  - EV charging management

---

## 5. Market Price & Trading

### epftoolbox - Electricity Price Forecasting
- **License**: AGPL-3.0 (⚠️ Copyleft - requires sharing modifications if distributed)
- **Repository**: `https://github.com/jeslago/epftoolbox`
- **Commercial Use**: Can use internally but must open-source modifications if distributed
- **Key Features**:
  - Benchmark models for price forecasting
  - Multiple market datasets included
  - Statistical and ML models implemented
  - Evaluation metrics and backtesting
- **Markets Covered**:
  - EPEX (Germany, France, Belgium)
  - NordPool (Nordic countries)
  - PJM (US Mid-Atlantic)
  - Historical data 2014-2020

---

## 6. Advanced Deep Learning Implementations

### Torch-NILM
- **License**: MIT License
- **Repository**: `https://github.com/Virtsionis/torch-nilm`
- **Commercial Use**: Complete freedom to modify and keep private
- **Key Features**:
  - PyTorch implementation of NILM algorithms
  - Standardized experiment framework
  - CNN, LSTM, Seq2Seq architectures
  - Automated hyperparameter tuning
  - Multi-GPU training support
- **Algorithms Implemented**:
  - Sequence-to-sequence models
  - Temporal convolutional networks
  - Attention mechanisms
  - Transfer learning capabilities

### V2G_TS_Project - Vehicle-to-Grid
- **License**: MIT License
- **Repository**: `https://github.com/sohaibdaoudi/V2G_TS_Project`
- **Commercial Use**: Full freedom to use and modify privately
- **Key Features**:
  - V2G optimization with deep learning
  - BiLSTM and GRU architectures
  - Demand response modeling
  - EV fleet management
  - Grid stability analysis
- **Components**:
  - Load forecasting module
  - Price prediction module
  - Optimization engine
  - Visualization dashboard

---

## 7. Additional Utility Projects

### ADGEfficiency/forecast
- **License**: MIT License
- **Repository**: `https://github.com/ADGEfficiency/forecast`
- **Key Features**:
  - Toolkit for energy time series forecasting
  - Naive baselines through deep learning
  - Feature engineering utilities
  - Cross-validation for time series

### lstm-load-forecasting
- **License**: MIT License (verify on repository)
- **Repository**: `https://github.com/dafrie/lstm-load-forecasting`
- **Key Features**:
  - LSTM implementation for load forecasting
  - Swiss electricity market focus
  - Keras/TensorFlow implementation
  - Hourly resolution predictions

### short-term-energy-demand-forecasting
- **License**: Check repository
- **Repository**: `https://github.com/kolasniwash/short-term-energy-demand-forecasting`
- **Key Features**:
  - 24-hour ahead forecasting
  - Multiple models: Prophet, SARIMA, Keras
  - Spanish electricity market data
  - Comparative benchmarks

---

## Recommended Implementation Strategy

### For Immediate Production Use:
1. **Grid-Scale Forecasting**: Start with OpenSTEF (MPL 2.0) or PyPSA (MIT)
2. **Baseline Estimation**: Implement CalTRACK/OpenEEmeter (Apache 2.0)
3. **Disaggregation**: Use NILMTK or Torch-NILM (Apache/MIT)

### For Research & Development:
1. **Datasets**: Download from HuggingFace or use NILMTK parsers
2. **Benchmarking**: Use epftoolbox (note AGPL license restrictions)
3. **Deep Learning**: Extend Torch-NILM (MIT) for custom models

### License Risk Mitigation:
- **Lowest Risk**: MIT-licensed projects (PyPSA, Torch-NILM, Energym)
- **Medium Risk**: Apache 2.0 (CalTRACK, NILMTK) - must acknowledge changes
- **Manageable Risk**: MPL 2.0 (OpenSTEF) - keep as separate service/library
- **Avoid for Products**: AGPL (epftoolbox) unless willing to open-source

### Data Privacy Considerations:
All listed projects (except AGPL-licensed ones) allow you to:
- Keep your energy consumption data completely private
- Run analyses without disclosing results
- Build proprietary models on top
- Sell services based on the analysis
- Modify algorithms without sharing improvements

---

## Contact & Updates
For the latest versions and additional projects, check:
- GitHub Topics: `energy-forecasting`, `nilm`, `smart-grid`
- HuggingFace Hub: Search for "energy" or "electricity"
- Papers with Code: Energy forecasting benchmarks