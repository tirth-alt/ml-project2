#!/usr/bin/env python3
"""
Predictive Maintenance for Rental Bike Fleets
Team: Anomaly Archers (Section B)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("="*60)
print("🚴 PREDICTIVE MAINTENANCE FOR RENTAL BIKE FLEETS")
print("   Team: Anomaly Archers")
print("="*60)

# ============================================================
# 1. DATA LOADING
# ============================================================
print("\n📥 Loading data...")

df_nov = pd.read_csv('data/raw/202411-capitalbikeshare-tripdata.csv')
df_dec = pd.read_csv('data/raw/202412-capitalbikeshare-tripdata.csv')
df = pd.concat([df_nov, df_dec], ignore_index=True)

print(f"✅ Loaded {len(df):,} trip records")
print(f"   November: {len(df_nov):,} trips")
print(f"   December: {len(df_dec):,} trips")

# ============================================================
# 2. DATA CLEANING
# ============================================================
print("\n🧹 Cleaning data...")

df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])
df['duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

# Filter valid trips
df_clean = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 180)].copy()
print(f"✅ After cleaning: {len(df_clean):,} trips (removed {len(df)-len(df_clean):,} invalid)")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n⚙️ Engineering features...")

# Aggregate by station
station_features = df_clean.groupby('start_station_id').agg(
    total_trips=('ride_id', 'count'),
    total_duration_min=('duration_min', 'sum'),
    avg_trip_duration=('duration_min', 'mean'),
    std_trip_duration=('duration_min', 'std'),
    max_trip_duration=('duration_min', 'max'),
    min_trip_duration=('duration_min', 'min'),
    first_trip=('started_at', 'min'),
    last_trip=('started_at', 'max'),
    unique_destinations=('end_station_id', 'nunique'),
    member_trips=('member_casual', lambda x: (x == 'member').sum()),
    casual_trips=('member_casual', lambda x: (x == 'casual').sum())
).reset_index()

# Derived features
station_features['days_active'] = (station_features['last_trip'] - station_features['first_trip']).dt.days + 1
station_features['trips_per_day'] = station_features['total_trips'] / station_features['days_active']
station_features['member_ratio'] = station_features['member_trips'] / station_features['total_trips']
station_features['duration_variability'] = station_features['std_trip_duration'] / station_features['avg_trip_duration']
station_features = station_features.fillna(0)

# Filter low-activity stations
min_trips = 50
station_features_filtered = station_features[station_features['total_trips'] >= min_trips].copy()
print(f"✅ Created features for {len(station_features_filtered)} stations (>={min_trips} trips)")

# Prepare feature matrix
feature_columns = [
    'total_trips', 'total_duration_min', 'avg_trip_duration', 'std_trip_duration',
    'trips_per_day', 'unique_destinations', 'member_ratio', 'duration_variability'
]
X = station_features_filtered[feature_columns].copy()
X = X.replace([np.inf, -np.inf], 0).fillna(0)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4. K-MEANS CLUSTERING
# ============================================================
print("\n🔵 Running K-Means clustering...")

# Find optimal K
inertias = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True)

axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('outputs/figures/elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()

# Train final model with K=3
K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
station_features_filtered['cluster'] = cluster_labels

# Analyze clusters
cluster_summary = station_features_filtered.groupby('cluster')[feature_columns].mean()
heavy_cluster = cluster_summary['trips_per_day'].idxmax()
light_cluster = cluster_summary['trips_per_day'].idxmin()
moderate_cluster = [c for c in range(K) if c not in [heavy_cluster, light_cluster]][0]

cluster_names = {
    heavy_cluster: '🔴 Heavy Usage',
    moderate_cluster: '🟡 Moderate Usage',
    light_cluster: '🟢 Light Usage'
}
station_features_filtered['cluster_name'] = station_features_filtered['cluster'].map(cluster_names)

print(f"✅ Clustering complete!")
for k, v in cluster_names.items():
    count = (station_features_filtered['cluster'] == k).sum()
    print(f"   Cluster {k} ({v}): {count} stations")

# Visualize clusters with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
colors = ['#2ecc71', '#f1c40f', '#e74c3c']
for cluster_id in sorted(station_features_filtered['cluster'].unique()):
    mask = station_features_filtered['cluster'] == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[cluster_id],
                label=cluster_names[cluster_id], alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('Station Clusters - Usage Profiles')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/figures/cluster_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 5. ANOMALY DETECTION
# ============================================================
print("\n🔴 Running Anomaly Detection (Isolation Forest)...")

iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
anomaly_labels = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.decision_function(X_scaled)

station_features_filtered['anomaly'] = anomaly_labels
station_features_filtered['anomaly_score'] = anomaly_scores
station_features_filtered['is_anomaly'] = station_features_filtered['anomaly'] == -1

print(f"✅ Anomaly detection complete!")
print(f"   Normal stations: {(anomaly_labels == 1).sum()}")
print(f"   Anomalous stations: {(anomaly_labels == -1).sum()}")

# Visualize anomalies
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[~station_features_filtered['is_anomaly'], 0],
            X_pca[~station_features_filtered['is_anomaly'], 1],
            c='#3498db', label='Normal', alpha=0.5, s=80)
plt.scatter(X_pca[station_features_filtered['is_anomaly'], 0],
            X_pca[station_features_filtered['is_anomaly'], 1],
            c='#e74c3c', label='Anomaly', alpha=0.9, s=150, marker='X', edgecolors='black', linewidth=1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Anomaly Detection Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/figures/anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 6. TIME SERIES ANALYSIS
# ============================================================
print("\n📈 Running Time Series Analysis...")

df_clean['date'] = df_clean['started_at'].dt.date
daily_stats = df_clean.groupby(['start_station_id', 'date']).agg(
    daily_trips=('ride_id', 'count'),
    daily_duration=('duration_min', 'sum'),
    avg_duration=('duration_min', 'mean')
).reset_index()

def calculate_trend(group):
    if len(group) < 7:
        return 0
    x = np.arange(len(group))
    y = group['daily_trips'].values
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope

station_trends = daily_stats.groupby('start_station_id').apply(calculate_trend).reset_index()
station_trends.columns = ['start_station_id', 'usage_trend']

station_features_filtered = station_features_filtered.merge(station_trends, on='start_station_id', how='left')
station_features_filtered['usage_trend'] = station_features_filtered['usage_trend'].fillna(0)

station_features_filtered['trend_category'] = pd.cut(
    station_features_filtered['usage_trend'],
    bins=[-np.inf, -0.5, 0.5, np.inf],
    labels=['📉 Declining', '➡️ Stable', '📈 Increasing']
)

print(f"✅ Trend analysis complete!")
for cat in ['📉 Declining', '➡️ Stable', '📈 Increasing']:
    count = (station_features_filtered['trend_category'] == cat).sum()
    print(f"   {cat}: {count} stations")

# Fleet daily trend
fleet_daily = df_clean.groupby('date').agg(total_trips=('ride_id', 'count')).reset_index()
fleet_daily['date'] = pd.to_datetime(fleet_daily['date'])
fleet_daily['rolling_7d'] = fleet_daily['total_trips'].rolling(7).mean()

plt.figure(figsize=(14, 6))
plt.plot(fleet_daily['date'], fleet_daily['total_trips'], alpha=0.3, label='Daily Trips')
plt.plot(fleet_daily['date'], fleet_daily['rolling_7d'], linewidth=2, color='red', label='7-Day Rolling Avg')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.title('Fleet-Wide Daily Trip Volume')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/figures/time_series_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 7. HEALTH SCORING
# ============================================================
print("\n🏥 Calculating Health Scores...")

# Cluster risk
cluster_risk_map = {heavy_cluster: 0.7, moderate_cluster: 0.4, light_cluster: 0.1}
station_features_filtered['cluster_risk'] = station_features_filtered['cluster'].map(cluster_risk_map)

# Normalize anomaly score
min_score = station_features_filtered['anomaly_score'].min()
max_score = station_features_filtered['anomaly_score'].max()
station_features_filtered['normalized_anomaly'] = 1 - ((station_features_filtered['anomaly_score'] - min_score) / (max_score - min_score))

# Trend risk - convert to string first to handle categorical
station_features_filtered['trend_category_str'] = station_features_filtered['trend_category'].astype(str)
trend_risk_map = {'📉 Declining': 0.8, '➡️ Stable': 0.3, '📈 Increasing': 0.1}
station_features_filtered['trend_risk'] = station_features_filtered['trend_category_str'].map(trend_risk_map).fillna(0.3)

# Composite health score
w_cluster, w_anomaly, w_trend = 0.4, 0.4, 0.2
station_features_filtered['health_score'] = (
    w_cluster * station_features_filtered['cluster_risk'] +
    w_anomaly * station_features_filtered['normalized_anomaly'] +
    w_trend * station_features_filtered['trend_risk']
)

# Health categories
def categorize_health(score):
    if score < 0.35:
        return '🟢 Stable'
    elif score < 0.55:
        return '🟡 Warning'
    else:
        return '🔴 Critical'

station_features_filtered['health_category'] = station_features_filtered['health_score'].apply(categorize_health)

print(f"✅ Health scoring complete!")
for cat in ['🟢 Stable', '🟡 Warning', '🔴 Critical']:
    count = (station_features_filtered['health_category'] == cat).sum()
    pct = count / len(station_features_filtered) * 100
    print(f"   {cat}: {count} stations ({pct:.1f}%)")

# ============================================================
# 8. PRIORITY RANKING
# ============================================================
print("\n🔧 Creating Maintenance Priority Ranking...")

priority_ranking = station_features_filtered.sort_values('health_score', ascending=False)[
    ['start_station_id', 'cluster_name', 'health_score', 'health_category',
     'total_trips', 'trips_per_day', 'is_anomaly', 'trend_category']
].reset_index(drop=True)
priority_ranking.index = priority_ranking.index + 1
priority_ranking.index.name = 'Priority'

# Save ranking
priority_ranking.to_csv('outputs/maintenance_priority_ranking.csv')
print(f"✅ Priority ranking saved to outputs/maintenance_priority_ranking.csv")

print("\n📋 TOP 20 STATIONS FOR MAINTENANCE:")
print(priority_ranking.head(20).to_string())

# ============================================================
# 9. FINAL VISUALIZATIONS
# ============================================================
print("\n📊 Creating final visualizations...")

# Health distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

health_counts = station_features_filtered['health_category'].value_counts()
color_map = {'🟢 Stable': '#2ecc71', '🟡 Warning': '#f1c40f', '🔴 Critical': '#e74c3c'}
colors = [color_map.get(c, '#95a5a6') for c in health_counts.index]
axes[0].bar(health_counts.index, health_counts.values, color=colors)
axes[0].set_xlabel('Health Category')
axes[0].set_ylabel('Number of Stations')
axes[0].set_title('Fleet Health Distribution')

axes[1].hist(station_features_filtered['health_score'], bins=20, edgecolor='black', alpha=0.7, color='#3498db')
axes[1].axvline(x=0.35, color='green', linestyle='--', label='Stable threshold')
axes[1].axvline(x=0.55, color='orange', linestyle='--', label='Warning threshold')
axes[1].set_xlabel('Health Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Health Score Histogram')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/figures/health_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Top 10 priority
top10 = priority_ranking.head(10)
plt.figure(figsize=(12, 6))
colors = ['#e74c3c' if cat == '🔴 Critical' else '#f1c40f' for cat in top10['health_category']]
plt.barh(range(len(top10)), top10['health_score'], color=colors)
plt.yticks(range(len(top10)), [f"Station {sid}" for sid in top10['start_station_id']])
plt.xlabel('Health Score (Higher = More Urgent)')
plt.title('🔧 Top 10 Stations Requiring Maintenance')
plt.gca().invert_yaxis()
for i, (score, cat) in enumerate(zip(top10['health_score'], top10['health_category'])):
    plt.text(score + 0.01, i, f'{score:.2f}', va='center')
plt.tight_layout()
plt.savefig('outputs/figures/top10_priority.png', dpi=150, bbox_inches='tight')
plt.close()

# Pie chart
plt.figure(figsize=(8, 8))
health_counts = station_features_filtered['health_category'].value_counts()
plt.pie(health_counts.values, labels=health_counts.index, autopct='%1.1f%%',
        colors=['#2ecc71', '#f1c40f', '#e74c3c'][:len(health_counts)],
        explode=[0.05] * len(health_counts), shadow=True, startangle=90)
plt.title('Overall Fleet Health Status')
plt.savefig('outputs/figures/fleet_health_pie.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n" + "="*60)
print("📊 PROJECT SUMMARY")
print("="*60)
print(f"\nTotal trips analyzed: {len(df_clean):,}")
print(f"Stations analyzed: {len(station_features_filtered)}")
print(f"\nCluster Distribution:")
for name in station_features_filtered['cluster_name'].unique():
    count = (station_features_filtered['cluster_name'] == name).sum()
    print(f"  {name}: {count} stations")
print(f"\nAnomaly Detection:")
print(f"  Anomalous stations: {station_features_filtered['is_anomaly'].sum()}")
print(f"  Anomaly rate: {station_features_filtered['is_anomaly'].mean()*100:.1f}%")
print(f"\nHealth Status:")
for cat in ['🟢 Stable', '🟡 Warning', '🔴 Critical']:
    count = (station_features_filtered['health_category'] == cat).sum()
    pct = count / len(station_features_filtered) * 100
    print(f"  {cat}: {count} stations ({pct:.1f}%)")
print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE!")
print("="*60)
print("\n📁 Outputs saved to:")
print("   - outputs/maintenance_priority_ranking.csv")
print("   - outputs/figures/elbow_silhouette.png")
print("   - outputs/figures/cluster_visualization.png")
print("   - outputs/figures/anomaly_detection.png")
print("   - outputs/figures/time_series_trend.png")
print("   - outputs/figures/health_distribution.png")
print("   - outputs/figures/top10_priority.png")
print("   - outputs/figures/fleet_health_pie.png")
