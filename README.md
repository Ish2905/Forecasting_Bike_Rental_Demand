# Bike Rental Demand Forecasting

## 📋 Project Overview
This project predicts hourly bike rental demand for the Capital Bikeshare program in Washington D.C. using machine learning models. The goal is to forecast bike rental counts based on weather conditions, time features, and other environmental factors.

**Dataset:** Kaggle Bike Sharing Demand Competition  
**Primary Metric:** Root Mean Squared Logarithmic Error (RMSLE)

## 🎯 What It Does
- Analyzes historical bike rental data with weather and temporal features
- Engineers features like peak hours, day of week, and month
- Trains multiple machine learning models (Linear Regression, Random Forest, Gradient Boosting, Decision Tree)
- Creates a stacking ensemble that combines top-performing models
- Generates predictions for test data ready for Kaggle submission

## 🔧 Approach

### 1. **Data Preprocessing**
- Extract time features (hour, day of week, month) from datetime
- Create peak hour indicators for weekday rush hours and weekend leisure times
- Remove multicollinear features (temp vs atemp correlation: 0.98)
- Remove season feature (high correlation with month: 0.97)

### 2. **Model Training**
- Train 4 base models with optimized hyperparameters
- Use 10-fold cross-validation for robust evaluation
- Evaluate using RMSLE, R², and MAE metrics

### 3. **Stacking Ensemble**
- Select top 3 models based on RMSLE performance
- Train meta-model (Linear Regression) on base model predictions
- Generate final predictions from the ensemble

## 🚀 Setup and Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/Ish2905/Forecasting_Bike_Rental_Demand.git
cd Forecasting_Bike_Rental_Demand
```

2. **Install required packages**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

3. **Verify data files**
Ensure the following files exist in the `data/` directory:
- `train.csv` - Training dataset
- `test.csv` - Test dataset
- `sampleSubmission.csv` - Submission format reference

## 📊 Running the Project

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook bike_rental_simple.ipynb
```
Run all cells sequentially to:
- Load and explore data
- Visualize correlations
- Preprocess features
- Train models
- Generate submission file

### Option 2: Python Script
```bash
python bike_rental_simple.py
```

## 📁 Project Structure
```
Forecasting_Bike_Rental_Demand/
├── bike_rental_simple.ipynb    # Main notebook with full pipeline
├── data/
│   ├── train.csv               # Training data
│   ├── test.csv                # Test data
│   └── sampleSubmission.csv    # Submission format
├── submission.csv              # Generated predictions (after running)
└── README.md                   # This file
```

## 📈 Results
The project trains and evaluates multiple models:
- **Linear Regression** - Baseline model
- **Random Forest** - Ensemble of decision trees
- **Gradient Boosting** - Sequential boosting algorithm
- **Decision Tree** - Single tree model
- **Stacking Ensemble** - Combines top 3 models

Performance is measured using:
- **RMSLE** (Primary metric) - Penalizes underestimation
- **R²** - Variance explained
- **MAE** - Average prediction error

## 📝 Output
Running the notebook generates:
- `submission.csv` - Predictions for Kaggle submission
- Correlation heatmap visualization
- Model performance comparison table
- Detailed evaluation metrics for all models

## 🛠️ Key Features
- Clean, well-documented code with comprehensive comments
- Correlation analysis to identify and remove multicollinear features
- Peak hour detection based on weekday/weekend patterns
- 10-fold cross-validation for reliable model evaluation
- Stacking ensemble for improved predictions

## 📚 Dependencies
- pandas - Data manipulation
- numpy - Numerical operations
- scikit-learn - Machine learning models and evaluation
- matplotlib - Plotting
- seaborn - Statistical visualizations
