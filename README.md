# ✈️ Voyage Analytics — Travel & Tourism ML System

> **Flight Price Prediction · Gender Classification · Hotel Recommendation**

A full end-to-end machine learning system built on travel and tourism data, covering regression, classification, and collaborative filtering — with an MLOps deployment stack using Flask, Docker, Kubernetes, Airflow, and Jenkins.

---

## 📋 Project Overview

The travel industry generates massive transactional data. This project leverages three datasets — **flights**, **hotels**, and **users** — to solve three core business problems:

| Task | Type | Best Model |
|---|---|---|
| Flight Price Prediction | Regression | Random Forest Regressor |
| User Gender Classification | Classification | Random Forest Classifier |
| Hotel Recommendation | Recommendation | Item-based Collaborative Filtering |

---

## 📁 Project Structure

```
voyage-analytics/
│
├── Voyage_Analytics.ipynb       # Main Jupyter notebook
│
├── models/
│   ├── flight_price_rf_model.pkl
│   ├── gender_classifier_rf.pkl
│   ├── scaler.pkl
│   ├── le_flight_type.pkl
│   └── le_agency.pkl
│
├── data/
│   ├── flights.csv              # 271,888 records
│   ├── hotels.csv
│   └── users.csv
│
└── README.md
```

---

## 📊 Datasets

| Dataset | Records | Key Columns |
|---|---|---|
| `flights.csv` | 271,888 | `userCode`, `flightType`, `price`, `distance`, `time`, `agency`, `date` |
| `hotels.csv` | — | `userCode`, `name`, `place`, `days`, `total`, `price` |
| `users.csv` | ~1,340 | `code`, `gender`, `age`, `company` |

Datasets are linked via `userCode` / `travelCode`.

---

## 🔧 Tech Stack

**Core ML & Data**
- Python 3, Pandas, NumPy, Scikit-learn, SciPy

**Experiment Tracking**
- MLflow — all runs, metrics, and model versions are tracked

**Visualization**
- Matplotlib, Seaborn

**Deployment (MLOps)**
- Flask REST API → Docker → Kubernetes
- Apache Airflow (workflow orchestration)
- Jenkins (CI/CD)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/voyage-analytics.git
cd voyage-analytics
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy mlflow joblib
```

### 3. Add datasets

Place `flights.csv`, `hotels.csv`, and `users.csv` inside the `data/` folder.

### 4. Run the notebook

Open and run `Voyage_Analytics.ipynb` end-to-end in Jupyter or Google Colab.

---

## 🤖 ML Models

### 1. Flight Price Prediction (Regression)

**Features used:** `flightType`, `agency`, `time`, `distance`, `month`, `dayofweek`

**Models evaluated:**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | Higher | Higher | Lower |
| Decision Tree | Medium | Medium | Medium |
| **Random Forest** ✅ | **Lowest** | **Lowest** | **Highest** |

**Key insight:** `distance` and `flightType` are the top predictors. First class flights are significantly more expensive than economic/premium.

Hyperparameter tuning was performed using `RandomizedSearchCV` (6 iterations, 3-fold CV).

---

### 2. Gender Classification

**Features used:** `age`, `company_enc`

**Target:** `gender` (male / female — `none` values excluded)

| Model | Accuracy |
|---|---|
| Logistic Regression | Baseline |
| **Random Forest** ✅ | **Best** |

The dataset is naturally balanced (~452 male, ~448 female), so no class imbalance handling was required.

---

### 3. Hotel Recommendation (Collaborative Filtering)

- Builds a **user–hotel interaction matrix** from booking history
- Applies **cosine similarity** on hotel vectors (item-based CF)
- Returns the top-N most similar hotels to a given hotel name

```python
recommend_hotels("Hotel Name", top_n=5)
```

---

## 📈 Hypothesis Testing

Three statistical tests were performed:

| Hypothesis | Test Used | Outcome |
|---|---|---|
| First class flights are pricier than economic | Independent t-test | ✅ Rejected H₀ |
| No gender-based price difference | Independent t-test | ✅ Failed to reject H₀ |
| Distance and price are correlated | Pearson correlation | ✅ Rejected H₀ |

---

## 💾 Saving & Loading Models

```python
import joblib

# Save
joblib.dump(rf, 'models/flight_price_rf_model.pkl')

# Load and predict
model = joblib.load('models/flight_price_rf_model.pkl')
predicted_price = model.predict(new_flight_df)[0]
```

---

## 🧪 Sanity Check

```python
new_flight = pd.DataFrame({
    'flightType_enc': [2],   # firstClass
    'agency_enc': [1],        # CloudFy
    'time': [3.5],
    'distance': [800.0],
    'month': [7],
    'dayofweek': [4]
})
print(f"Predicted Price: ${model.predict(new_flight)[0]:.2f}")
```

---

## 📌 Key Business Insights

- **Pricing strategy:** First class pricing has the highest revenue potential; dynamic pricing can exploit seasonal peaks (summer/holiday months).
- **Corporate travel:** Top companies represent bulk-booking opportunities for B2B deals.
- **Extended stays:** Hotel bundle deals for longer stays can increase average booking value.
- **Agency optimization:** Agencies with lower median prices attract price-sensitive travelers.

---

## 🔮 Future Work

- Deploy the Flask REST API and containerize with Docker
- Set up Kubernetes for auto-scaling under load
- Automate retraining pipelines with Apache Airflow
- Integrate CI/CD with Jenkins
- Add XGBoost / LightGBM for improved regression performance
- Build a Streamlit or Gradio frontend for interactive predictions

---

## 👤 Author

**Individual Project** — Machine Learning Capstone

---

## 📄 License

This project is for educational purposes.
