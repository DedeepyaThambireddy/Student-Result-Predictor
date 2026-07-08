# 📖 Student Result Prediction System - Will I Pass?

A web application that helps students check whether they are likely to **pass or fail** their exams based on their study habits and academic record.

Built with **Python** and **Streamlit**, powered by a **Logistic Regression** classifier trained on real student data.

---

## 🖥️ Demo Preview

| Home Page | Predict Page | Data Analysis |
|---|---|---|
| Student-friendly landing page explaining how to use the tool | Sliders for 5 inputs → instant PASS / FAIL result with tips | Interactive charts showing class-wide patterns |

---

## 📁 Project Structure

```
student-result-prediction/
│
├── data_cleaning.py          # Step 1 — Load raw data, remove duplicates, check types
├── add_result.py             # Step 2 — Compute weighted score and add Result column
├── train_model.py            # Step 3 — Train Logistic Regression, save .pkl files
├── evaluate_model.py         # Step 4 — Load saved model and print evaluation metrics
├── app.py                    # Step 5 — Streamlit web application
│
├── student_dataset_balanced_.xlsx   # Raw input dataset
├── student_dataset.xlsx             # Cleaned dataset with Result column (auto-generated)
├── student_model.pkl                # Saved trained model (auto-generated)
├── scaler.pkl                       # Saved StandardScaler (auto-generated)
│
└── README.md
```

---

## ⚙️ How the Pipeline Works

The project follows a clean, modular structure where each script handles exactly one step:

```
student_dataset_balanced_.xlsx
        │
        ▼
 data_cleaning.py         →  removes duplicates, validates data types
        │
        ▼
   add_result.py          →  computes weighted score, adds Result (0/1)
        │
        ▼
  train_model.py          →  trains model, saves student_model.pkl + scaler.pkl
        │
        ▼
 evaluate_model.py        →  loads .pkl files, prints Accuracy / Precision / Recall / F1
        │
        ▼
      app.py              →  Streamlit UI — predictions, charts, model metrics
```

---

## 🔢 Input Features

| Feature | Range | Description |
|---|---|---|
| `Study_Hours` | 0 – 12 | Average hours studied per day |
| `Attendance` | 0 – 100% | Percentage of classes attended |
| `Previous_Marks` | 0 – 100 | Score in the previous exam |
| `Assignments` | 0 – 10 | Number of assignments submitted |
| `Sleep_Hours` | 4 – 10 | Average sleep hours per night |
| `Result` *(target)* | 0 / 1 | 0 = Fail, 1 = Pass (score ≥ 70) |

The `Result` column is computed by `add_result.py` using this weighted formula:

```python
score = (
    Study_Hours     * 4    +
    Attendance      * 0.30 +
    Previous_Marks  * 0.50 +
    Assignments     * 2    +
    (8 - abs(Sleep_Hours - 7)) * 2 +
    random noise (Gaussian, σ=5)
)
Result = 1 if score >= 70 else 0
```

---

## 📊 Model Performance

Evaluated on 100 held-out student records (20% test split, `random_state=42`):

| Metric | Score |
|---|---|
| Accuracy | **91.00%** |
| Precision | **89.47%** |
| Recall | **94.44%** |
| F1 Score | **91.89%** |

> **Why Recall matters here:** A high recall (94.44%) means the model successfully identifies nearly all students who are genuinely at risk — very few slip through without being flagged.

---

## 🌐 Application Pages

The Streamlit app has **4 pages**, accessible from the sidebar:

### 🏠 Home
- Introduction to the tool written for students
- Step-by-step guide on how to use it
- Overview of the 5 input factors

### 🔮 Predict My Result
- Adjust 5 sliders for your personal inputs
- Instant **PASS / FAIL** result with confidence percentage
- Live **Readiness Score** (0–100) updates as you move sliders
- If at risk → **personalised tips** targeting your weakest areas
- **Radar chart** comparing your profile to the average passing student

### 📊 Data Analysis
- Scatter plots: Study Hours vs Marks, Attendance vs Marks
- Pass / Fail donut chart and Study Hours histogram (pass vs fail overlay)
- Box plots: Attendance and Assignments by outcome
- Full **Feature Correlation Heatmap**
- Raw dataset browser

### 📈 Model Performance
- Colour-coded metric cards (Accuracy, Precision, Recall, F1)
- **Confusion Matrix** heatmap with plain-English interpretation
- **Feature Importance** bar chart showing which factors matter most
- Plain-English metric guide for non-technical readers

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit 1.35+ | Web application framework |
| scikit-learn 1.4+ | Logistic Regression, train/test split, metrics |
| Pandas 2.0+ | Data loading and manipulation |
| NumPy 1.26+ | Numerical operations |
| joblib 1.3+ | Saving and loading `.pkl` model files |
| Plotly 5.20+ | Interactive charts |
| openpyxl | Reading and writing `.xlsx` Excel files |

---

## 🔮 Future Improvements

-  Connect to a live database (MySQL / PostgreSQL) instead of Excel files
-  Compare multiple models — Random Forest, SVM, XGBoost
-  Deploy publicly on Streamlit Cloud
-  Add role-based access — Student / Teacher / Admin dashboards
-  Automated email alerts for at-risk students
-  Export individual prediction reports as PDF

---
