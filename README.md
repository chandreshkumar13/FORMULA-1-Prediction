# 🏎️ Formula 1 Race Prediction System

An AI/ML-based Formula 1 analytics and race prediction project built using real F1 telemetry data from FastF1.

This project analyzes:
- Driver performance
- Lap times
- Tire strategies
- Race consistency
- Fastest laps

and uses Machine Learning to predict race outcomes.

---

# 🚀 Features

✅ Real Formula 1 telemetry data  
✅ Multiple race analysis  
✅ Driver comparison graphs  
✅ Automatic dataset generation  
✅ Feature engineering  
✅ Random Forest ML model  
✅ CSV dataset export  
✅ Tire compound analysis  

---

# 🛠️ Technologies Used

- Python
- FastF1 API
- Pandas
- Matplotlib
- Scikit-learn

---

# 📂 Project Structure

```bash
F1-Race-Predictor/
│
├── cache/
├── f1_dataset.csv
├── main.py
├── requirements.txt
├── README.md

# ⚙️ How It Works

This project uses real Formula 1 race data from the FastF1 API.

### Step 1: Data Collection
The system fetches real race telemetry data from Formula 1 races such as:
- Monaco GP
- Silverstone GP
- Monza GP

### Step 2: Data Processing
The collected data is cleaned and transformed by:
- Removing missing lap times
- Converting lap times into seconds
- Organizing driver performance data

### Step 3: Feature Engineering
For every driver, the project calculates:

- Average Lap Time
- Fastest Lap
- Consistency Score
- Total Laps Completed
- Most Used Tire Compound

### Step 4: Dataset Creation
All driver statistics are stored in a dataset and exported as:

```bash
f1_dataset.csv
```

### Step 5: Machine Learning
The dataset is used to train a Random Forest Classifier.

The model learns patterns from driver performance data and predicts race outcomes.

### Step 6: Visualization
The project generates graphs comparing drivers such as:

- VER vs LEC Lap Times
- Performance Trends
- Race Pace Analysis

---

# ▶️ How To Run

### 1. Clone Repository

```bash
git clone https://github.com/chandreshkumar13/FORMULA-1-Prediction.git
```

### 2. Move Into Project Folder

```bash
cd FORMULA-1-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run The Project

```bash
python main.py
```

### 5. Output

The program will:

- Download F1 race data
- Generate driver statistics
- Create f1_dataset.csv
- Train a Machine Learning model
- Display prediction results
- Show driver comparison graphs

---

# Sample Workflow

```text
FastF1 API
    ↓
Race Data
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Dataset Creation
    ↓
Machine Learning
    ↓
Predictions
    ↓
Visualization
```