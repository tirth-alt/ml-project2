#!/usr/bin/env python3
"""
Data Synthesis Script
Adds synthetic bike_id, odometer readings, user_rating, complaint_flag to the dataset
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚴 DATA SYNTHESIS FOR BIKE-LEVEL ANALYSIS")
print("="*60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n📥 Loading data...")
df_nov = pd.read_csv('data/raw/202411-capitalbikeshare-tripdata.csv')
df_dec = pd.read_csv('data/raw/202412-capitalbikeshare-tripdata.csv')
df = pd.concat([df_nov, df_dec], ignore_index=True)
print(f"✅ Loaded {len(df):,} trips")

# ============================================================
# 2. BASIC CLEANING
# ============================================================
print("\n🧹 Cleaning data...")
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])
df['duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

# Filter valid trips
df = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 180)].copy()
print(f"✅ After cleaning: {len(df):,} trips")

# ============================================================
# 3. SYNTHESIZE BIKE_ID
# ============================================================
print("\n🔧 Synthesizing bike_id...")

# Fleet configuration
CLASSIC_BIKES = 2000
ELECTRIC_BIKES = 3000
np.random.seed(42)

# Separate by bike type
df_classic = df[df['rideable_type'] == 'classic_bike'].copy()
df_electric = df[df['rideable_type'] == 'electric_bike'].copy()

print(f"   Classic bikes trips: {len(df_classic):,}")
print(f"   Electric bikes trips: {len(df_electric):,}")

def assign_bike_ids(df_subset, fleet_size, prefix):
    """Assign bike IDs with realistic uneven distribution"""
    n_trips = len(df_subset)
    weights = np.random.pareto(1.5, fleet_size) + 1
    weights = weights / weights.sum()
    bike_indices = np.random.choice(fleet_size, size=n_trips, p=weights)
    bike_ids = [f"{prefix}_{str(i).zfill(4)}" for i in bike_indices]
    return bike_ids

df_classic['bike_id'] = assign_bike_ids(df_classic, CLASSIC_BIKES, 'CLASSIC')
df_electric['bike_id'] = assign_bike_ids(df_electric, ELECTRIC_BIKES, 'EBIKE')

# Combine back
df = pd.concat([df_classic, df_electric], ignore_index=True)
df = df.sort_values('started_at').reset_index(drop=True)

print(f"✅ Created {df['bike_id'].nunique():,} unique bike IDs")

# ============================================================
# 4. SYNTHESIZE ODOMETER READINGS (Distance Meter)
# ============================================================
print("\n📏 Synthesizing odometer readings...")

# Each bike has a starting odometer value and accumulates distance per trip
# Trip distance = end_odometer - start_odometer

# Initialize starting odometer for each bike (varying ages of bikes)
bike_initial_odometer = {}
for bike_id in df['bike_id'].unique():
    # New bikes: 0-500 km, Medium: 500-2000 km, Old: 2000-5000 km
    age_category = np.random.choice(['new', 'medium', 'old'], p=[0.2, 0.5, 0.3])
    if age_category == 'new':
        bike_initial_odometer[bike_id] = np.random.uniform(0, 500)
    elif age_category == 'medium':
        bike_initial_odometer[bike_id] = np.random.uniform(500, 2000)
    else:
        bike_initial_odometer[bike_id] = np.random.uniform(2000, 5000)

print(f"   Initialized odometers for {len(bike_initial_odometer)} bikes")

# For each trip, generate realistic trip distance based on duration
# Average speed ~12 km/h for bikes, with variation
def generate_trip_distance(duration_min):
    """Generate realistic trip distance based on duration"""
    # Base speed: 10-15 km/h with variation
    avg_speed = np.random.uniform(8, 16)  # km/h
    base_distance = (duration_min / 60) * avg_speed
    # Add some noise
    noise = np.random.uniform(0.8, 1.2)
    return max(0.1, base_distance * noise)  # Minimum 0.1 km

# Generate trip distances
df['trip_distance_km'] = df['duration_min'].apply(generate_trip_distance)

# Calculate cumulative odometer per bike (sorted by time)
df = df.sort_values(['bike_id', 'started_at']).reset_index(drop=True)

# Calculate start and end odometer for each trip
start_odometers = []
end_odometers = []
current_odometer = {}

for _, row in df.iterrows():
    bike_id = row['bike_id']
    distance = row['trip_distance_km']
    
    if bike_id not in current_odometer:
        current_odometer[bike_id] = bike_initial_odometer[bike_id]
    
    start_odo = current_odometer[bike_id]
    end_odo = start_odo + distance
    
    start_odometers.append(round(start_odo, 2))
    end_odometers.append(round(end_odo, 2))
    
    current_odometer[bike_id] = end_odo

df['start_odometer_km'] = start_odometers
df['end_odometer_km'] = end_odometers

# Recalculate trip_distance from odometer (for consistency)
df['trip_distance_km'] = df['end_odometer_km'] - df['start_odometer_km']

print(f"✅ Odometer readings generated")
print(f"   Max odometer: {df['end_odometer_km'].max():.1f} km")
print(f"   Average trip distance: {df['trip_distance_km'].mean():.2f} km")

# Resort by started_at for final output
df = df.sort_values('started_at').reset_index(drop=True)

# ============================================================
# 5. SYNTHESIZE USER_RATING (Nullable - most users don't rate)
# ============================================================
print("\n⭐ Synthesizing user_rating...")

rating_probability = 0.15
n_trips = len(df)
gets_rating = np.random.random(n_trips) < rating_probability

# Rating distribution: 5★ (40%), 4★ (35%), 3★ (15%), 2★ (7%), 1★ (3%)
rating_weights = [0.03, 0.07, 0.15, 0.35, 0.40]
ratings = np.random.choice([1, 2, 3, 4, 5], size=n_trips, p=rating_weights)

df['user_rating'] = np.where(gets_rating, ratings, np.nan)

rated_count = df['user_rating'].notna().sum()
print(f"✅ User ratings: {rated_count:,} rated ({rated_count/n_trips*100:.1f}%)")
print(f"   Average rating: {df['user_rating'].mean():.2f} ⭐")

# ============================================================
# 6. SYNTHESIZE COMPLAINT_FLAG
# ============================================================
print("\n🚨 Synthesizing complaint_flag...")

base_complaint_prob = 0.02
short_trip_boost = np.where(df['duration_min'] < 3, 0.05, 0)
low_rating_boost = np.where(df['user_rating'] <= 2, 0.10, 0)
low_rating_boost = np.where(df['user_rating'].isna(), 0, low_rating_boost)
complaint_prob = np.clip(base_complaint_prob + short_trip_boost + low_rating_boost, 0, 0.20)
df['complaint_flag'] = (np.random.random(n_trips) < complaint_prob).astype(int)

complaint_count = df['complaint_flag'].sum()
print(f"✅ Complaints: {complaint_count:,} ({complaint_count/n_trips*100:.2f}%)")

# ============================================================
# 7. SYNTHESIZE DAYS_SINCE_SERVICE
# ============================================================
print("\n🔧 Synthesizing days_since_service...")

bike_service_days = {}
for bike_id in df['bike_id'].unique():
    if np.random.random() < 0.3:
        bike_service_days[bike_id] = np.random.randint(0, 30)
    else:
        bike_service_days[bike_id] = np.random.randint(30, 180)

df['days_since_service'] = df['bike_id'].map(bike_service_days)
print(f"✅ Service days range: {df['days_since_service'].min()}-{df['days_since_service'].max()}")

# ============================================================
# 8. SAVE ENHANCED DATASET
# ============================================================
print("\n💾 Saving enhanced dataset...")

output_cols = [
    'ride_id', 'bike_id', 'rideable_type', 'started_at', 'ended_at',
    'duration_min', 'start_odometer_km', 'end_odometer_km', 'trip_distance_km',
    'start_station_id', 'end_station_id',
    'start_lat', 'start_lng', 'end_lat', 'end_lng',
    'member_casual', 'user_rating', 'complaint_flag', 'days_since_service'
]

df_output = df[output_cols].copy()
df_output.to_csv('data/processed/enhanced_trips.csv', index=False)

print(f"✅ Saved to data/processed/enhanced_trips.csv")
print(f"   Rows: {len(df_output):,}")
print(f"   Columns: {len(output_cols)}")

# ============================================================
# 9. SUMMARY
# ============================================================
print("\n" + "="*60)
print("📊 SYNTHESIS SUMMARY")
print("="*60)
print(f"\nTotal trips: {len(df):,}")
print(f"Unique bikes: {df['bike_id'].nunique():,}")
print(f"  - Classic: {df[df['rideable_type']=='classic_bike']['bike_id'].nunique():,}")
print(f"  - Electric: {df[df['rideable_type']=='electric_bike']['bike_id'].nunique():,}")
print(f"\nNew columns added:")
print(f"  ✅ bike_id - Synthetic unique bike identifier")
print(f"  ✅ start_odometer_km - Odometer reading at trip start")
print(f"  ✅ end_odometer_km - Odometer reading at trip end")
print(f"  ✅ trip_distance_km - Distance from odometer (end - start)")
print(f"  ✅ user_rating - Synthetic (nullable, 15% rated)")
print(f"  ✅ complaint_flag - Synthetic (2% complaint rate)")
print(f"  ✅ days_since_service - Synthetic maintenance age")
print("\n" + "="*60)
print("✅ DATA SYNTHESIS COMPLETE!")
print("="*60)
