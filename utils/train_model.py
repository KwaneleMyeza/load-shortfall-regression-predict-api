"""
    GM 5 Random forest model
"""

# Dependencies
import pandas as pd
import pickle
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed', 'Valencia_wind_speed']]

# Fit model
rf_regression = RandomForestRegressor()
print ("Training Model...")
rf_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = 'C:/Users/nassa/OneDrive/Documents/Github/load-shortfall-regression-predict-api/assets/trained-models/load_shortfall_RF.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(rf_regression, open(save_path,'wb'))

