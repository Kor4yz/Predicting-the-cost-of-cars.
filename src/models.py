from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

def ridge_pipeline(transformer):
    return Pipeline([("prep", transformer), ("model", RidgeCV(alphas=[0.1,1,10]))])

def xgb_pipeline(transformer):
    return Pipeline([("prep", transformer),
                     ("model", XGBRegressor(n_estimators=600, max_depth=8,
                                            learning_rate=0.05, subsample=0.8,
                                            colsample_bytree=0.8, n_jobs=-1,
                                            tree_method="hist"))])
