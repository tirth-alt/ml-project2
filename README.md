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
