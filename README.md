# 🌤️ CloudCue — Intelligent Weather Forecasting

> **Catch the sky's next move.**  
> An intelligent end-to-end weather forecasting web application powered by Machine Learning, physical atmospheric modeling, and real-time Open-Meteo API telemetry.

---

## 🌟 Key Features

* **Real-Time Telemetry**: Fetches live weather observations (temperature, relative humidity, barometric pressure, wind speed, precipitation) for any global city via Open-Meteo.
* **Physics-Informed Feature Engineering**: Computes domain-specific meteorological features:
  * **Dew Point & Dew Point Spread**: Quantifies atmospheric moisture saturation.
  * **3-Hour Barometric Pressure Tendency ($\Delta P / \Delta t$)**: Detects incoming cyclonic depressions and storm fronts.
  * **Cyclical Encodings**: Smooth diurnal ($\sin / \cos$ hour) and seasonal ($\sin / \cos$ month) transformations.
  * **Temporal Weather Lags**: Historical lags ($t-1, t-3, t-6, t-24$) across temperature, humidity, pressure, wind, and precipitation.
* **Multi-Horizon ML Forecasts**:
  * **+1 Hour Horizon**: Next-hour temperature, humidity, and rainfall predictions.
  * **+3 Hour & +6 Hour Horizons**: Extended multi-horizon ML outlook.
* **Uncertainty Quantification**: 95% Confidence Intervals ($\pm 1.96 \times \text{RMSE}$) computed from test residuals.
* **Environmental & Biometeorology Advisories**:
  * **Steadman Heat Index & Wind Chill**: Calculates dynamic "Feels Like" thermal comfort categories.
  * **Storm & Cloudburst Threat Score (0–100)**: Evaluates pressure drops, dew point spread, and moisture instability.
  * **Fog / Visibility Warning**: Detects radiation and advection fog risk.
* **Live MLOps & Prediction Tracking**: SQLite-backed tracker that records every live prediction and evaluates real-world model error against verified observations.
* **Model Explainability**: Interactive feature importance attributions visualizing what atmospheric variables drive each prediction.
* **Streamlit Dark UI**: Interactive dashboard with responsive Plotly charts, advisory banners, and live MLOps metrics.

---

## 📊 Model Benchmark Summary

| Target | Deployed Best Model | Benchmark Metric | Score |
| :--- | :--- | :--- | :--- |
| **Temperature (+1h)** | XGBoost Regressor | $R^2$ / RMSE | **0.9731** / **$0.57^\circ\text{C}$** |
| **Humidity (+1h)** | XGBoost Regressor | $R^2$ / RMSE | **0.9696** / **$3.32\%$** |
| **Rainfall (+1h)** | Random Forest Classifier | Accuracy / $F_1$ | **$94.5\%$** / **$0.7588$** |
| **Temperature (+3h)** | XGBoost Regressor | $R^2$ / RMSE | **0.9024** / **$1.09^\circ\text{C}$** |
| **Temperature (+6h)** | XGBoost Regressor | $R^2$ / RMSE | **0.8666** / **$1.27^\circ\text{C}$** |

---

## 📁 Project Architecture

```
cloudcue/
├── app.py                   # Streamlit web application dashboard
├── data_collection.py       # Ingests historical weather archive data from Open-Meteo
├── preprocessing.py         # Data cleaning & feature calculation pipeline
├── train_model.py           # Model training, multi-horizon evaluation, & export
├── prediction.py            # Terminal-based live prediction CLI tool
├── weather_api.py           # Open-Meteo live API integration & real-time feature adapter
├── requirements.txt         # Project dependencies
├── utils/
│   ├── __init__.py
│   ├── features.py          # Canonical physical & cyclical feature definitions (DRY)
│   ├── advisory.py          # Heat index, wind chill, storm threat & fog detection
│   └── tracker.py           # SQLite MLOps prediction logging & error tracking
├── datasets/
│   ├── weather_data.csv            # Raw historical observations
│   ├── processed_weather_data.csv  # Feature-engineered training set
│   ├── model_evaluation.csv        # Multi-model evaluation benchmark metrics
│   ├── feature_importance.csv      # Feature importance percentages
│   └── forecast_tracker.db         # SQLite live prediction & drift log
└── models/
    ├── best_temperature_model.pkl
    ├── best_humidity_model.pkl
    ├── best_rainfall_model.pkl
    ├── best_temperature_3h_model.pkl
    ├── best_rainfall_3h_model.pkl
    ├── best_temperature_6h_model.pkl
    ├── best_rainfall_6h_model.pkl
    └── model_metadata.json          # Metrics, confidence interval bounds, & feature lists
```

---

## 🚀 Getting Started

### 1. Environment Setup

Create and activate a virtual environment:
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run Data Ingestion & Preprocessing (Optional if files exist)

```bash
python data_collection.py
python preprocessing.py
```

### 3. Train Models

Train the regression, classification, and multi-horizon models:
```bash
python train_model.py
```

### 4. Run the Streamlit Dashboard

Launch the interactive UI:
```bash
streamlit run app.py
```

### 5. Run Terminal CLI Inference

```bash
python prediction.py
```

---

## 🔬 Tech Stack
* **Language**: Python 3.10+
* **Machine Learning**: Scikit-Learn, XGBoost, Joblib
* **Data & Analytics**: Pandas, NumPy
* **Visualization & Frontend**: Streamlit, Plotly Express
* **APIs**: Open-Meteo Forecast & Archive APIs

