# Toy script to train a simple regression model for traffic congestion forecasting.
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rng = np.random.RandomState(42)
X = []
y = []
for day in range(60):
    for hour in range(24):
        base = 0.4 + 0.4 * (1 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0)
        temp = 20 + 10 * np.sin((day/30.0)*2*np.pi)
        noise = rng.normal(scale=0.05)
        congestion = min(1.0, max(0.0, base + 0.1*np.sin(hour/24.0*2*np.pi) + 0.02*(temp-20) + noise))
        X.append([hour, day%7, temp])
        y.append(congestion)
X = np.array(X); y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('RMSE:', mean_squared_error(y_test, preds, squared=False))
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Saved model to model.pkl')
