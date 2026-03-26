# 💸 ExpenseIQ

ML-based personal expense tracking and forecasting system.

## 🚀 Features
- Upload bank statements (multi-bank compatible)
- Automatic expense extraction (debits only)
- Smart categorization (Food, Shopping, etc.)
- ML-based monthly expense forecasting
- Income-aware prediction constraints
- Interactive dashboard (Streamlit)

## 🧠 ML Approach
- Time-series forecasting on daily expenses
- Feature engineering:
  - Lag values
  - Rolling averages
  - Seasonal features
- Model selection:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- Validation using MAE

## 📊 Tech Stack
- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn
- Plotly

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py