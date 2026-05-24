import os
import fastf1
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================================
# CREATE CACHE FOLDER
# =========================================

if not os.path.exists('cache'):
    os.makedirs('cache')

# Enable cache
fastf1.Cache.enable_cache('cache')

# =========================================
# LOAD MULTIPLE RACES
# =========================================

races = ['Monaco', 'Silverstone', 'Monza']

all_laps = pd.DataFrame()

for race in races:

    print(f"\nLoading {race} GP...")

    session = fastf1.get_session(2024, race, 'R')
    session.load()

    race_laps = session.laps

    # Add race name
    race_laps['Race'] = race

    # Combine all races
    all_laps = pd.concat([all_laps, race_laps])

# Final combined dataset
laps = all_laps

# =========================================
# AUTOMATIC DRIVER DATASET GENERATION
# =========================================

drivers = laps['Driver'].unique()

dataset = []

for driver in drivers:

    # Filter driver laps
    driver_laps = laps[laps['Driver'] == driver]

    # Remove missing lap times
    driver_laps = driver_laps.dropna(subset=['LapTime'])

    # Skip empty data
    if len(driver_laps) == 0:
        continue

    # Convert lap time to seconds
    driver_laps['LapSeconds'] = (
        driver_laps['LapTime']
        .dt.total_seconds()
    )

    # =====================================
    # FEATURES
    # =====================================

    avg_lap = driver_laps['LapSeconds'].mean()

    consistency = driver_laps['LapSeconds'].std()

    fastest_lap = driver_laps['LapSeconds'].min()

    total_laps = len(driver_laps)

    # Most used tire compound
    main_compound = (
        driver_laps['Compound']
        .mode()[0]
    )

    # =====================================
    # STORE DATA
    # =====================================

    dataset.append({
        'Driver': driver,
        'AverageLap': avg_lap,
        'Consistency': consistency,
        'FastestLap': fastest_lap,
        'TotalLaps': total_laps,
        'MainCompound': main_compound
    })

# =========================================
# CREATE DATAFRAME
# =========================================

df = pd.DataFrame(dataset)

print("\n===== DRIVER DATASET =====")
print(df)

# =========================================
# SAVE CSV
# =========================================

df.to_csv('f1_dataset.csv', index=False)

print("\nDataset saved successfully!")

# =========================================
# CREATE LABELS
# =========================================

# Example:
# Winner = 1
# Others = 0

df['Winner'] = df['Driver'].apply(
    lambda x: 1 if x == 'LEC' else 0
)

print("\n===== DATASET WITH LABELS =====")
print(df)

# =========================================
# ENCODE TIRE COMPOUND
# =========================================

df['MainCompound'] = (
    df['MainCompound']
    .astype('category')
    .cat.codes
)

# =========================================
# MACHINE LEARNING
# =========================================

# Features
X = df[
    [
        'AverageLap',
        'Consistency',
        'FastestLap',
        'TotalLaps',
        'MainCompound'
    ]
]

# Labels
y = df['Winner']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n===== MODEL RESULTS =====")
print("Predictions:", predictions)
print("Accuracy:", accuracy)

# =========================================
# DRIVER COMPARISON GRAPH
# =========================================

# Compare Verstappen and Leclerc

ver_laps = laps[laps['Driver'] == 'VER']
ver_laps = ver_laps.dropna(subset=['LapTime'])

ver_laps['LapSeconds'] = (
    ver_laps['LapTime']
    .dt.total_seconds()
)

lec_laps = laps[laps['Driver'] == 'LEC']
lec_laps = lec_laps.dropna(subset=['LapTime'])

lec_laps['LapSeconds'] = (
    lec_laps['LapTime']
    .dt.total_seconds()
)

# Plot graph
plt.plot(
    ver_laps['LapNumber'],
    ver_laps['LapSeconds'],
    label='VER'
)

plt.plot(
    lec_laps['LapNumber'],
    lec_laps['LapSeconds'],
    label='LEC'
)

# Labels
plt.xlabel('Lap Number')
plt.ylabel('Lap Time (seconds)')
plt.title('VER vs LEC Lap Comparison')

# Legend
plt.legend()

# Show graph
plt.show()