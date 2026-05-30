================================================================================
         HOUSE PRICE PREDICTION — KAGGLE COMPETITION
================================================================================

R2 Score : 0.9176
RMSE     : $25,141
Models   : XGBoost + GBR + Ridge (Voting Ensemble)
Language : Python 3.8+

Predict residential home sale prices using advanced regression techniques —
featuring ensemble modeling, feature engineering, and hyperparameter tuning.

--------------------------------------------------------------------------------
TABLE OF CONTENTS
--------------------------------------------------------------------------------
  1. Overview
  2. Dataset
  3. Project Structure
  4. Installation
  5. Workflow
  6. Feature Engineering
  7. Models Used
  8. Results
  9. How to Run
 10. Future Improvements

--------------------------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------------------------
This project is based on the Kaggle House Prices: Advanced Regression Techniques
competition. The goal is to predict the final sale price of homes in Ames, Iowa,
using 79 explanatory variables describing almost every aspect of residential homes.

Kaggle Link:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Final Model : Voting Regressor (Ensemble of GBR + XGBoost + Ridge)
R2 Score    : 0.9176
RMSE        : $25,141

--------------------------------------------------------------------------------
2. DATASET
--------------------------------------------------------------------------------
Source       : https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
Train set    : 1,460 rows x 81 columns
Test set     : 1,459 rows x 80 columns
Target var   : SalePrice

NOTE: Download train.csv and test.csv from Kaggle and place them in the root
directory before running the notebook.

--------------------------------------------------------------------------------
3. PROJECT STRUCTURE
--------------------------------------------------------------------------------
house-price-prediction/
|
|-- train.csv          -> Training data (download from Kaggle)
|-- test.csv           -> Test data (download from Kaggle)
|-- Housing_data_prediction.ipynb    -> Main Jupyter Notebook
|-- README.txt         -> Project documentation

--------------------------------------------------------------------------------
4. INSTALLATION
--------------------------------------------------------------------------------
# Clone the repository
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn xgboost

--------------------------------------------------------------------------------
5. WORKFLOW
--------------------------------------------------------------------------------
  1. Data Loading
       |
  2. Exploratory Data Analysis (EDA)
       |
  3. Missing Value Treatment
       |
  4. Feature Engineering
       |
  5. Preprocessing Pipeline (Imputer + Scaler + Encoder)
       |
  6. Model Training with GridSearchCV
       |
  7. Ensemble (VotingRegressor)
       |
  8. Evaluation (RMSE + R2)

--------------------------------------------------------------------------------
6. FEATURE ENGINEERING
--------------------------------------------------------------------------------
New features created from existing columns:

  Feature             Formula                                      Reason
  ------------------- -------------------------------------------- -------------------
  houseage            YrSold - YearBuilt                           Age of house
  houserermodelage    YrSold - YearRemodAdd                        Age since remodel
  totalsf             1stFlrSF+2ndFlrSF+BsmtFinSF1+BsmtFinSF2     Total finished area
  totalarea           GrLivArea + TotalBsmtSF                      Total living space
  totalbaths          FullBath+0.5*HalfBath+BsmtFullBath+          Weighted bath count
                      0.5*BsmtHalfBath
  totalporchsf        OpenPorchSF + 3SsnPorch + EnclosedPorch      Total porch area

Columns Dropped:
  - High null columns  : PoolQC, MiscFeature, Alley, Fence
  - Redundant columns  : GarageArea, GarageYrBlt, GarageCond, BsmtFinType2
  - Replaced by above  : YrSold, YearBuilt, YearRemodAdd, 1stFlrSF, 2ndFlrSF, etc.

--------------------------------------------------------------------------------
7. MODELS USED
--------------------------------------------------------------------------------
Preprocessing Pipeline:
  Numerical columns    -> SimpleImputer(mean)          -> StandardScaler
  Ordinal categorical  -> SimpleImputer(most_frequent) -> OrdinalEncoder
  Nominal categorical  -> SimpleImputer(most_frequent) -> OneHotEncoder

Individual Models (tuned with GridSearchCV):
  Model                        Tuned Parameters
  ---------------------------- ----------------------------------------
  Linear Regression            baseline
  Ridge Regression             alpha, solver
  Random Forest Regressor      max_depth, n_estimators, min_samples_split
  XGBoost Regressor            learning_rate, max_depth, gamma, subsample
  Gradient Boosting Regressor  learning_rate, max_depth, n_estimators

Final Model — VotingRegressor:
  ('gbr',   GBR_cv.best_estimator_)    weight = 2
  ('xgb',   xgb_cv.best_estimator_)    weight = 3  <- highest (best performer)
  ('ridge', ridge_cv.best_estimator_)  weight = 1

--------------------------------------------------------------------------------
8. RESULTS
--------------------------------------------------------------------------------
  RMSE  :  25,141.21
  R2    :  0.9176

  - Model explains 91.76% of the variance in house prices
  - Average prediction error ~$25,141 on a typical $200k home (~12.5% error)

--------------------------------------------------------------------------------
9. HOW TO RUN
--------------------------------------------------------------------------------
  Step 1 : Download train.csv and test.csv from Kaggle (link in section 2)
  Step 2 : Place both files in the same folder as the notebook
  Step 3 : Open Housing_data_prediction.ipynb in Jupyter
  Step 4 : Run all cells top to bottom (Kernel -> Restart & Run All)

--------------------------------------------------------------------------------
10. FUTURE IMPROVEMENTS
--------------------------------------------------------------------------------
  [ ] Apply log transform on SalePrice to reduce skewness (expected RMSE: ~$18k)
  [ ] Remove outliers in LotArea and LotFrontage
  [ ] Try Stacking instead of simple Voting ensemble
  [ ] Add LightGBM to the ensemble
  [ ] Generate submission.csv for Kaggle leaderboard
--------------------------------------------------------------------------------

