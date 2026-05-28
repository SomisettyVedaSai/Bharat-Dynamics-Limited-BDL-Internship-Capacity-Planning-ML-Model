# Bharat Dynamics Limited (BDL) - Internship Capacity Planning & Analytics Suite

An executive-grade Machine Learning & Data Analytics dashboard developed for capacity planning, bottleneck prediction, and scheduling risk assessment at Bharat Dynamics Limited (BDL), a Government of India Enterprise under the Ministry of Defence.

## 🚀 Key Features

*   **Model-less Side-by-Side Planning Dashboard**: Overhauled v2 layout explicitly mapping operator inputs to AI planning outcomes in a glassmorphic slate dark theme.
*   **Capacity Clearance Certificate**: A printable, secure A4 report featuring input parameters, utilization ratings, overload severity badges, SHAP explainability drivers, and PSU sign-off blocks.
*   **Routing Sequence Simulator**: Real-time simulation of production runs across all chronological work centers to identify capacity bottlenecks dynamically.
*   **Stress Testing**: Real-time simulation data visualization for extreme production quantities and shift patterns.
*   **Explainable AI (SHAP)**: Fully integrated Shapley value attributions displaying the top 3 physical drivers for every capacity bottleneck.
*   **Predictions Log Database**: View, load, or delete past planning trials (supports automatic form re-population and re-runs on click).

## 🛠️ Technology Stack

*   **Frontend**: HTML5, Vanilla CSS3 (Custom styling, modern glassmorphism, responsive grid layouts), JavaScript (AJAX/Fetch APIs, Plotly.js charts).
*   **Backend**: Python 3.13, Flask (RESTful web services, templates integration).
*   **Database**: MongoDB (Local/Fallback storage modes).
*   **Machine Learning**: Scikit-Learn (Gradient Boosting Regressors/Classifiers), SHAP (Explainable AI values).

---

## 💻 Setup & Installation

### 1. Prerequisites
*   Python 3.10+
*   MongoDB (optional; system will automatically fall back to local file storage if MongoDB is not running)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Mock Data & Train Models
```bash
# Generate synthetic production data
python data_generation.py

# Train models A, B, C, D, and E
python train_pipeline.py
```

### 4. Run the Dashboard
```bash
python app.py
```
Open [http://127.0.0.1:5050](http://127.0.0.1:5050) in your web browser.

---

## 🔒 Security Statement
*This project has been developed under strict air-gap design parameters. External content delivery networks (CDNs) and public assets are completely bypassed on print clearance certificates to protect sensitive internal planning information.*
