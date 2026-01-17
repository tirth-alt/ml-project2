#!/usr/bin/env python3
"""
Bike-Level Predictive Maintenance Analysis
Uses synthesized bike_id and new features
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("="*60)
print("🚴 BIKE-LEVEL PREDICTIVE MAINTENANCE ANALYSIS")
print("   Team: Anomaly Archers")
print("="*60)

# ============================================================
# 1. LOAD ENHANCED DATA
# ============================================================
print("\n📥 Loading enhanced data...")

df = pd.read_csv('data/processed/enhanced_trips.csv')
df['started_at'] = pd.to_datetime(df['started_at'])

print(f"✅ Loaded {len(df):,} trips")
print(f"   Unique bikes: {df['bike_id'].nunique():,}")

# ============================================================
# 2. CREATE BIKE-LEVEL FEATURES
# ============================================================
print("\n⚙️ Creating bike-level features...")

bike_features = df.groupby('bike_id').agg(
    rideable_type=('rideable_type', 'first'),
    total_trips=('ride_id', 'count'),
    total_duration_min=('duration_min', 'sum'),
    avg_trip_duration=('duration_min', 'mean'),
    std_trip_duration=('duration_min', 'std'),
    total_distance_km=('trip_distance_km', 'sum'),
    avg_trip_distance=('trip_distance_km', 'mean'),
    first_trip=('started_at', 'min'),
    last_trip=('started_at', 'max'),
    unique_start_stations=('start_station_id', 'nunique'),
    unique_end_stations=('end_station_id', 'nunique'),
    member_trips=('member_casual', lambda x: (x == 'member').sum()),
    # New synthesized features
    avg_user_rating=('user_rating', 'mean'),
    rating_count=('user_rating', 'count'),
    ratings_given=('user_rating', lambda x: x.notna().sum()),
    complaint_count=('complaint_flag', 'sum'),
    days_since_service=('days_since_service', 'first')  # Same per bike
).reset_index()

# Derived features
bike_features['days_active'] = (bike_features['last_trip'] - bike_features['first_trip']).dt.days + 1
bike_features['trips_per_day'] = bike_features['total_trips'] / bike_features['days_active']
bike_features['member_ratio'] = bike_features['member_trips'] / bike_features['total_trips']
bike_features['complaint_rate'] = bike_features['complaint_count'] / bike_features['total_trips']
bike_features['short_trip_ratio'] = df.groupby('bike_id').apply(
    lambda x: (x['duration_min'] < 3).sum() / len(x)
).values

# Fill NaN
bike_features = bike_features.fillna(0)

print(f"✅ Created features for {len(bike_features):,} bikes")
print(f"\nFeature preview:")
print(bike_features[['bike_id', 'total_trips', 'total_distance_km', 'avg_user_rating', 'complaint_count']].head())

# ============================================================
# 3. PREPARE FEATURES FOR ML
# ============================================================
print("\n📊 Preparing ML features...")

feature_columns = [
    'total_trips',
    'total_distance_km',
    'avg_trip_duration',
    'avg_trip_distance',
    'trips_per_day',
    'member_ratio',
    'avg_user_rating',
    'complaint_rate',
    'short_trip_ratio',
    'days_since_service',
    'cumulative_mileage'  # From odometer
]

# Add cumulative mileage (max end_odometer for each bike)
max_odometer = df.groupby('bike_id')['end_odometer_km'].max().reset_index()
max_odometer.columns = ['bike_id', 'cumulative_mileage']
bike_features = bike_features.merge(max_odometer, on='bike_id', how='left')

X = bike_features[feature_columns].copy()
X = X.replace([np.inf, -np.inf], 0).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Feature matrix: {X_scaled.shape}")

# ============================================================
# 4. K-MEANS CLUSTERING
# ============================================================
print("\n🔵 Running K-Means clustering...")

# Find optimal K
inertias = []
silhouette_scores_list = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores_list.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True)

axes[1].plot(K_range, silhouette_scores_list, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('K')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('outputs/figures/bike_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()

# Train with K=4 (better for bike types)
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
bike_features['cluster'] = cluster_labels

print(f"✅ Clustering complete (K={K})")
print(f"\nCluster distribution:")
print(bike_features['cluster'].value_counts().sort_index())

# Name clusters based on characteristics
cluster_summary = bike_features.groupby('cluster')[feature_columns].mean()

# Identify cluster types
usage_order = cluster_summary['trips_per_day'].sort_values().index.tolist()
cluster_names = {
    usage_order[0]: '🟢 Light Usage',
    usage_order[1]: '🟡 Moderate Usage',
    usage_order[2]: '🟠 Heavy Usage',
    usage_order[3]: '🔴 Extreme Usage'
}
bike_features['cluster_name'] = bike_features['cluster'].map(cluster_names)

print(f"\nCluster naming:")
for c, name in cluster_names.items():
    count = (bike_features['cluster'] == c).sum()
    print(f"  {name}: {count} bikes")

# Visualize
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
for cluster_id in sorted(bike_features['cluster'].unique()):
    mask = bike_features['cluster'] == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[cluster_id],
                label=cluster_names[cluster_id], alpha=0.6, s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Bike Clusters by Usage Profile')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/figures/bike_cluster_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 5. ANOMALY DETECTION
# ============================================================
print("\n🔴 Running Anomaly Detection...")

iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
anomaly_labels = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.decision_function(X_scaled)

bike_features['anomaly'] = anomaly_labels
bike_features['anomaly_score'] = anomaly_scores
bike_features['is_anomaly'] = anomaly_labels == -1

print(f"✅ Anomaly detection complete")
print(f"   Normal bikes: {(anomaly_labels == 1).sum()}")
print(f"   Anomalous bikes: {(anomaly_labels == -1).sum()}")

# Visualize
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[~bike_features['is_anomaly'], 0],
            X_pca[~bike_features['is_anomaly'], 1],
            c='#3498db', label='Normal', alpha=0.5, s=50)
plt.scatter(X_pca[bike_features['is_anomaly'], 0],
            X_pca[bike_features['is_anomaly'], 1],
            c='#e74c3c', label='Anomaly', alpha=0.9, s=100, marker='X')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Bike Anomaly Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/figures/bike_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 6. TIME SERIES TRENDS
# ============================================================
print("\n📈 Running Time Series Analysis...")

df['date'] = df['started_at'].dt.date
daily_stats = df.groupby(['bike_id', 'date']).agg(
    daily_trips=('ride_id', 'count'),
    daily_distance=('trip_distance_km', 'sum')
).reset_index()

def calculate_trend(group):
    if len(group) < 7:
        return 0
    x = np.arange(len(group))
    y = group['daily_trips'].values
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope

bike_trends = daily_stats.groupby('bike_id').apply(calculate_trend).reset_index()
bike_trends.columns = ['bike_id', 'usage_trend']

bike_features = bike_features.merge(bike_trends, on='bike_id', how='left')
bike_features['usage_trend'] = bike_features['usage_trend'].fillna(0)

bike_features['trend_category'] = pd.cut(
    bike_features['usage_trend'],
    bins=[-np.inf, -0.3, 0.3, np.inf],
    labels=['📉 Declining', '➡️ Stable', '📈 Increasing']
)
bike_features['trend_category_str'] = bike_features['trend_category'].astype(str)

print(f"✅ Trend analysis complete")
print(f"\nTrend distribution:")
print(bike_features['trend_category'].value_counts())

# ============================================================
# 7. HEALTH SCORING
# ============================================================
print("\n🏥 Calculating Health Scores...")

# Component 1: Cluster risk
cluster_risk_map = {
    usage_order[0]: 0.1,  # Light
    usage_order[1]: 0.3,  # Moderate
    usage_order[2]: 0.6,  # Heavy
    usage_order[3]: 0.9   # Extreme
}
bike_features['cluster_risk'] = bike_features['cluster'].map(cluster_risk_map)

# Component 2: Anomaly risk (normalized 0-1)
min_score = bike_features['anomaly_score'].min()
max_score = bike_features['anomaly_score'].max()
bike_features['normalized_anomaly'] = 1 - ((bike_features['anomaly_score'] - min_score) / (max_score - min_score))

# Component 3: Trend risk
trend_risk_map = {'📉 Declining': 0.7, '➡️ Stable': 0.3, '📈 Increasing': 0.1}
bike_features['trend_risk'] = bike_features['trend_category_str'].map(trend_risk_map).fillna(0.3)

# Component 4: Complaint risk
bike_features['complaint_risk'] = (bike_features['complaint_rate'] * 10).clip(0, 1)

# Component 5: Service age risk
bike_features['service_risk'] = (bike_features['days_since_service'] / 180).clip(0, 1)

# Component 6: Rating risk (low ratings = high risk)
bike_features['rating_risk'] = np.where(
    bike_features['avg_user_rating'] > 0,
    1 - (bike_features['avg_user_rating'] / 5),
    0.5  # Default if no ratings
)

# Composite health score
weights = {
    'cluster_risk': 0.20,
    'normalized_anomaly': 0.20,
    'trend_risk': 0.10,
    'complaint_risk': 0.20,
    'service_risk': 0.15,
    'rating_risk': 0.15
}

bike_features['health_score'] = (
    weights['cluster_risk'] * bike_features['cluster_risk'] +
    weights['normalized_anomaly'] * bike_features['normalized_anomaly'] +
    weights['trend_risk'] * bike_features['trend_risk'] +
    weights['complaint_risk'] * bike_features['complaint_risk'] +
    weights['service_risk'] * bike_features['service_risk'] +
    weights['rating_risk'] * bike_features['rating_risk']
)

# Categories
def categorize_health(score):
    if score < 0.35:
        return '🟢 Stable'
    elif score < 0.55:
        return '🟡 Warning'
    else:
        return '🔴 Critical'

bike_features['health_category'] = bike_features['health_score'].apply(categorize_health)

print(f"✅ Health scoring complete")
print(f"\nHealth distribution:")
for cat in ['🟢 Stable', '🟡 Warning', '🔴 Critical']:
    count = (bike_features['health_category'] == cat).sum()
    pct = count / len(bike_features) * 100
    print(f"  {cat}: {count} bikes ({pct:.1f}%)")

# ============================================================
# 8. PRIORITY RANKING
# ============================================================
print("\n🔧 Creating Priority Ranking...")

priority_ranking = bike_features.sort_values('health_score', ascending=False)[
    ['bike_id', 'rideable_type', 'cluster_name', 'health_score', 'health_category',
     'total_trips', 'total_distance_km', 'complaint_count', 'avg_user_rating',
     'days_since_service', 'is_anomaly']
].reset_index(drop=True)
priority_ranking.index = priority_ranking.index + 1
priority_ranking.index.name = 'Priority'

priority_ranking.to_csv('outputs/bike_maintenance_priority.csv')
print(f"✅ Saved to outputs/bike_maintenance_priority.csv")

print("\n📋 TOP 20 BIKES FOR MAINTENANCE:")
print(priority_ranking.head(20).to_string())

# ============================================================
# 9. VISUALIZATIONS
# ============================================================
print("\n📊 Creating visualizations...")

# Health distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

health_counts = bike_features['health_category'].value_counts()
color_map = {'🟢 Stable': '#2ecc71', '🟡 Warning': '#f1c40f', '🔴 Critical': '#e74c3c'}
bar_colors = [color_map.get(c, '#95a5a6') for c in health_counts.index]
axes[0].bar(health_counts.index, health_counts.values, color=bar_colors)
axes[0].set_xlabel('Health Category')
axes[0].set_ylabel('Number of Bikes')
axes[0].set_title('Fleet Health Distribution')

axes[1].hist(bike_features['health_score'], bins=30, edgecolor='black', alpha=0.7, color='#3498db')
axes[1].axvline(x=0.35, color='green', linestyle='--', label='Stable threshold')
axes[1].axvline(x=0.55, color='orange', linestyle='--', label='Warning threshold')
axes[1].set_xlabel('Health Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Health Score Distribution')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/figures/bike_health_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Top 10 priority
top10 = priority_ranking.head(10)
plt.figure(figsize=(12, 6))
bar_colors = ['#e74c3c' if cat == '🔴 Critical' else '#f1c40f' for cat in top10['health_category']]
plt.barh(range(len(top10)), top10['health_score'], color=bar_colors)
plt.yticks(range(len(top10)), top10['bike_id'])
plt.xlabel('Health Score (Higher = More Urgent)')
plt.title('🔧 Top 10 Bikes Requiring Maintenance')
plt.gca().invert_yaxis()
for i, score in enumerate(top10['health_score']):
    plt.text(score + 0.01, i, f'{score:.2f}', va='center')
plt.tight_layout()
plt.savefig('outputs/figures/bike_top10_priority.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance (correlation with health score)
correlations = bike_features[feature_columns + ['health_score']].corr()['health_score'].drop('health_score').sort_values()
plt.figure(figsize=(10, 6))
correlations.plot(kind='barh', color=['#e74c3c' if x > 0 else '#2ecc71' for x in correlations])
plt.xlabel('Correlation with Health Score')
plt.title('Feature Importance (Correlation Analysis)')
plt.tight_layout()
plt.savefig('outputs/figures/bike_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n" + "="*60)
print("📊 BIKE-LEVEL ANALYSIS SUMMARY")
print("="*60)
print(f"\nTotal bikes analyzed: {len(bike_features):,}")
print(f"  - Classic: {(bike_features['rideable_type']=='classic_bike').sum():,}")
print(f"  - Electric: {(bike_features['rideable_type']=='electric_bike').sum():,}")

print(f"\nCluster Distribution:")
for name in bike_features['cluster_name'].unique():
    count = (bike_features['cluster_name'] == name).sum()
    print(f"  {name}: {count} bikes")

print(f"\nAnomaly Detection:")
print(f"  Anomalous bikes: {bike_features['is_anomaly'].sum()} ({bike_features['is_anomaly'].mean()*100:.1f}%)")

print(f"\nHealth Status:")
for cat in ['🟢 Stable', '🟡 Warning', '🔴 Critical']:
    count = (bike_features['health_category'] == cat).sum()
    pct = count / len(bike_features) * 100
    print(f"  {cat}: {count} bikes ({pct:.1f}%)")

print("\n" + "="*60)
print("✅ BIKE-LEVEL ANALYSIS COMPLETE!")
print("="*60)
print("\n📁 Outputs:")
print("   - outputs/bike_maintenance_priority.csv")
print("   - outputs/figures/bike_*.png")

# Save bike features for further analysis
bike_features.to_csv('data/processed/bike_features.csv', index=False)
print("   - data/processed/bike_features.csv")
