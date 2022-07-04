from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler


class Model:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = XGBRegressor(
            base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bynode=1, colsample_bytree=0.7, gamma=0.0, gpu_id=-1,
            importance_type='gain', interaction_constraints='',
            learning_rate=0.1, max_delta_step=0, max_depth=12,
            min_child_weight=7, monotone_constraints='()',
            n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=42,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
            tree_method='exact', validate_parameters=1, verbosity=None)

    def fit(self, X, y):
        X_transformed = self.scaler.fit_transform(X)
        self.model = self.model.fit(X_transformed, y)

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


