from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from preprocessing import processdata

def split_data(data,target):
  x=data.drop(target)
  y=data[target]
  return x,y

def split_train_test_dataset(x,y):
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
  return x_train,x_test,y_train,y_test

def ml_forecasting_model(x_train, y_train, x_test, y_test):
rf_grid = {
    'n_estimators': [100, 300],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestRegressor()
grid_rf = GridSearchCV(estimator=rf_model, param_grid=rf_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_rf.fit(x_train, y_train)
best_rf = grid_rf.best_estimator_
print("✅ Best RF params:", grid_rf.best_params_)

# --- XGBoost Grid Search ---
xgb_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.7, 1.0]
}

xgb_model = XGBRegressor(objective='reg:squarederror')
grid_xgb = GridSearchCV(estimator=xgb_model, param_grid=xgb_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_xgb.fit(x_train, y_train)
best_xgb = grid_xgb.best_estimator_
print("✅ Best XGBoost params:", grid_xgb.best_params_)

# =======================================
# Step 3: Combine Models with VotingRegressor
# =======================================
voting_model = VotingRegressor(estimators=[
    ('xgb', best_xgb),
    ('rf', best_rf)
])

voting_model.fit(x_train, y_train)

# =======================================
# Step 4: Evaluation & Visualization
# =======================================
preds = voting_model.predict(x_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
return mae, rmse
