import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import neptune
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA, tSNE, KernelPCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV, SelectFromModel
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression, ElasticNet, Lars, Lasso, LinearRegression, OrthogonalMatchingPursuit, PassiveAggressiveRegressor, RANSACRegressor, Ridge, SGDRegressor, TheilSenRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.semi_supervised import SelfTrainingRegressor

# Load stock prices data
stock_prices = pd.read_csv('stock_prices.csv')

# Define feature engineering functions
def calculate_momentum(df):
    return df['price'].pct_change()

def calculate_volatility(df):
    return df['price'].rolling(window=30).std()

def calculate_correlation(df, asset):
    return df['price'].corr(asset['price'])

def calculate_macd(df):
    return df['price'].ewm(span=12, adjust=False).mean() - df['price'].ewm(span=26, adjust=False).mean()

def calculate_rsi(df):
    delta = df['price'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean().abs()
    rs = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_bollinger_bands(df):
    return df['price'].rolling(window=20).mean() + 2*df['price'].rolling(window=20).std()

def calculate_force_index(df):
    return df['price'].diff() * df['volume']

# Calculate features
stock_prices['momentum'] = calculate_momentum(stock_prices)
stock_prices['volatility'] = calculate_volatility(stock_prices)
stock_prices['correlation'] = calculate_correlation(stock_prices, pd.read_csv('asset_prices.csv'))
stock_prices['macd'] = calculate_macd(stock_prices)
stock_prices['rsi'] = calculate_rsi(stock_prices)
stock_prices['bollinger_bands'] = calculate_bollinger_bands(stock_prices)
stock_prices['force_index'] = calculate_force_index(stock_prices)

# Split data into training and testing sets
X = stock_prices.drop(['price', 'date'], axis=1)
y = stock_prices['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing pipeline
preprocessing_pipeline = ColumnTransformer([
    ('scaler', StandardScaler(), ['momentum', 'volatility', 'correlation', 'acd', 'rsi', 'bollinger_bands', 'force_index']),
    ('imputer', SimpleImputer(strategy='mean'), ['momentum', 'volatility', 'correlation', 'acd', 'rsi', 'bollinger_bands', 'force_index'])
])

# Define feature selection pipeline
feature_selection_pipeline = SelectKBest(f_classif, k=5)

# Define model pipeline
model_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('feature_selection', feature_selection_pipeline),
    ('model', VotingRegressor([
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
        ('cat', CatBoostRegressor(n_estimators=100, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=100, random_state=42)),
        ('br', BayesianRidge()),
        ('ard', ARDRegression()),
        ('en', ElasticNet()),
        ('lars', Lars()),
        ('lasso', Lasso()),
        ('lr', LinearRegression()),
        ('omp', OrthogonalMatchingPursuit()),
        ('pa', PassiveAggressiveRegressor()),
        ('ransac', RANSACRegressor()),
        ('ridge', Ridge()),
        ('sgd', SGDRegressor()),
        ('tsr', TheilSenRegressor()),
        ('svr', SVR()),
        ('nusvr', NuSVR()),
        ('lsvr', LinearSVR()),
        ('gpr', GaussianProcessRegressor()),
        ('dtr', DecisionTreeRegressor()),
        ('etr', ExtraTreeRegressor()),
        ('knn', KNeighborsRegressor()),
        ('rnr', RadiusNeighborsRegressor()),
        ('mlp', MLPRegressor()),
        ('ir', IsotonicRegression()),
        ('mor', MultiOutputRegressor()),
        ('str', SelfTrainingRegressor())
    ]))
])

# Define hyperparameter tuning pipeline
hyperparameter_tuning_pipeline = GridSearchCV(model_pipeline, {
    'odel__n_estimators': [100, 200, 300],
    'odel__max_depth': [None, 5, 10],
    'odel__learning_rate': [0.1, 0.5, 1]
}, cv=5, scoring='neg_mean_squared_error')

# Train the model
hyperparameter_tuning_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = hyperparameter_tuning_pipeline.predict(X_test)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log metrics to Neptune
neptune.log_metric('RMSE', rmse)
neptune.log_metric('MAPE', mape)
neptune.log_metric('R2', r2)

# Plot trend
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

# Define a function to calculate the Fear & Greed Index
def calculate_fear_greed_index():
    # Calculate the 7 indicators with the same weight
    indicators = [
        market_momentum(),
        stock_price_strength(),
        stock_price_breadth(),
        put_call_option_ratio(),
        junk_bond_demand(),
        market_volatility(),
        safe_haven_demand()
    ]
    return np.mean(indicators)

# Define a function to calculate the AI Risk Indicator
def calculate_ai_risk_indicator():
    # Calculate the directional movement of the stock market (momentum)
    momentum = calculate_momentum(stock_prices)
    
    # Calculate the risk of the stock market (volatility)
    volatility = calculate_volatility(stock_prices)
    
    # Calculate the relationship between the stock market and major asset groups (correlation)
    correlation = calculate_correlation(stock_prices, pd.read_csv('asset_prices.csv'))
    
    # Calculate the AI Risk Indicator score
    score = (momentum + volatility + correlation) / 3
    return score

# Define a function to generate the insurance plan
def generate_insurance_plan(score):
    if score > 50:
        return 'Risk Off: Take off risk in the portfolio'
    else:
        return 'Risk On: Invest in the stock market'

# Calculate the AI Risk Indicator score
score = calculate_ai_risk_indicator()

# Generate the insurance plan
insurance_plan = generate_insurance_plan(score)

print(insurance_plan)
