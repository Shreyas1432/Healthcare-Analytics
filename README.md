# Healthcare-Analytics
Analytics system for identifying healthcare capacity bottlenecks
## Overview
Irish hospitals struggle with recurring capacity constraints and unpredictable demand spikes, forcing administrators into reactive crisis management. This project forecasts healthcare bottlenecks 12 months in advance by analyzing 15 years of procedure volumes and wait times, enabling hospitals to pre-allocate resources, optimize staffing, and reduce patient wait times before problems occur.

## What I Built
- End-to-end data pipeline merging procedure counts and wait time data across 150,000+ records
- Feature engineering layer creating temporal indicators (growth rates, volatility, lagged metrics) to prevent data leakage
- Predictive analytics system generating area-specific bottleneck forecasts and resource allocation recommendations
- Automated reporting workflow producing CSV outputs and visualizations for operational decision-making

## Data & Processing
- **Source:** Irish public healthcare datasets (Procedures.csv, Waits.csv) spanning 2005-2019
- **Processing:** Data cleaning, temporal aggregation, feature engineering with rolling windows, train-test split maintaining temporal integrity
- **Output:** Integrated dataset with bottleneck scores, classification/regression results, 26,438 high-risk predictions, and geographic resource priorities

## Tech Stack
- Python (pandas, numpy, scikit-learn)
- SQL-like data transformations
- Matplotlib/Seaborn for visualization

## Key Outcomes
- Identified 26,438 high-risk area-procedure combinations for 2018-2019 requiring immediate resource intervention
- Revealed geographic disparities with Dublin/Cork accounting for 47% of bottleneck cases despite 35% of procedures
- Produced area-level resource priority rankings enabling targeted staffing and equipment allocation
- Generated temporal trend analysis showing 40% procedure volume increase and capacity stress patterns

## How to Run
```bash
pip install -r requirements.txt
python cleaned_IrishHealthcareAnalysis.py
