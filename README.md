# 🚴 Predictive Maintenance for Rental Bike Fleets

## Project Overview
This project uses **unsupervised machine learning** to predict which rental bike stations require maintenance, based on usage pattern analysis.

**Team**: Anomaly Archers (Section B)

## Techniques Used
- **K-Means Clustering**: Segment stations by usage profiles
- **Isolation Forest**: Detect anomalous usage patterns
- **Time Series Analysis**: Track usage degradation trends
- **Health Scoring**: Prioritize maintenance queue

## Project Structure
```
AML_Project_Anomaly_Archers/
├── data/
│   ├── raw/              # Original CSV files
│   └── processed/        # Cleaned data
├── notebooks/
│   └── main_analysis.ipynb   # Main analysis notebook
├── outputs/
│   ├── figures/          # Generated visualizations
│   └── maintenance_priority_ranking.csv
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Analysis
```bash
cd notebooks
jupyter notebook main_analysis.ipynb
```

### 3. View Results
- Health scores and priority rankings in `outputs/`
- Visualizations in `outputs/figures/`

## Dataset
- **Source**: Capital Bikeshare System Data (Washington D.C.)
- **Period**: November-December 2024
- **Size**: ~700,000+ trip records

## Key Outputs
1. **Cluster Analysis**: Stations grouped into Heavy/Moderate/Light usage
2. **Anomaly Detection**: Flagged stations with unusual patterns
3. **Health Scores**: 0-1 score for each station
4. **Priority Ranking**: Ordered maintenance queue

## Authors
- Mohit Kumar
- Krishna Faujdar
- Tirth Shah (Team Leader)
- Abhi Gandhi

---

# Section 2: Bike-Level Analysis

## Overview
Since the original dataset lacks individual bike IDs, we **synthesized** realistic bike identifiers to enable **bike-level** predictive maintenance analysis.

## New Files Created

| File | Purpose |
|------|---------|
| `src/synthesize_data.py` | Generates synthetic bike_id, odometer, ratings |
| `run_bike_analysis.py` | Main ML analysis script |
| `notebooks/bike_analysis.ipynb` | Interactive notebook |
| `data/processed/enhanced_trips.csv` | Trips with bike_id & odometer |
| `data/processed/bike_features.csv` | Bike-level aggregated features |
| `outputs/bike_maintenance_priority.csv` | Priority ranking by bike |

## Data Synthesis

### Synthetic Features Added

| Feature | Description |
|---------|-------------|
| `bike_id` | Unique identifier (CLASSIC_0001, EBIKE_0001, etc.) |
| `start_odometer_km` | Distance meter reading at trip start |
| `end_odometer_km` | Distance meter reading at trip end |
| `trip_distance_km` | end_odometer - start_odometer |
| `user_rating` | Customer rating 1-5 (nullable, 15% rate) |
| `complaint_flag` | Customer complaint (0/1, ~2.6% rate) |
| `days_since_service` | Days since last maintenance (0-179) |

### Fleet Configuration
- **Total bikes**: 5,000
  - Classic bikes: 2,000
  - Electric bikes: 3,000
- **Total trips**: 912,957

## ML Pipeline

```
Raw Trips (912K) → Synthesize bike_id → Aggregate to bike-level (5,000)
                                              ↓
                    ┌───────────────┬─────────────────┬──────────────┐
                    │   K-Means     │  Isolation      │  Time Series │
                    │   (K=4)       │  Forest (5%)    │  Trends      │
                    └───────┬───────┴────────┬────────┴──────┬───────┘
                            └────────────────┼───────────────┘
                                             ↓
                                    Health Score (0-1)
                                             ↓
                                    Priority Ranking
```

## Features Used (11 total)

1. `total_trips` - Number of trips per bike
2. `total_distance_km` - Total distance traveled
3. `avg_trip_duration` - Average trip length
4. `avg_trip_distance` - Average distance per trip
5. `trips_per_day` - Usage intensity
6. `member_ratio` - Member vs casual ratio
7. `avg_user_rating` - Average customer rating
8. `complaint_rate` - Percentage of trips with complaints
9. `short_trip_ratio` - Ratio of very short trips (<3 min)
10. `days_since_service` - Maintenance age
11. `cumulative_mileage` - Total odometer reading

## Health Score Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| Cluster Risk | 20% | Higher for extreme usage clusters |
| Anomaly Score | 20% | From Isolation Forest |
| Trend Risk | 10% | Declining usage = higher risk |
| Complaint Risk | 20% | Based on complaint rate |
| Service Risk | 15% | Based on days since service |
| Rating Risk | 15% | Low ratings = higher risk |

## Results

| Metric | Value |
|--------|-------|
| Bikes Analyzed | 5,000 |
| Clusters | 4 (Light, Moderate, Heavy, Extreme) |
| Anomalies | 250 (5%) |
| 🟢 Stable | 4,493 (89.9%) |
| 🟡 Warning | 499 (10%) |
| 🔴 Critical | 8 (0.2%) |

## How to Run Bike-Level Analysis

```bash
# Step 1: Synthesize data
python3 src/synthesize_data.py

# Step 2: Run analysis
python3 run_bike_analysis.py

# OR use notebook
jupyter notebook notebooks/bike_analysis.ipynb
```

## Output Files

- `outputs/bike_maintenance_priority.csv` - Priority ranking
- `outputs/figures/bike_*.png` - Visualizations
