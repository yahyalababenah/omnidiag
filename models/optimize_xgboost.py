import pandas as pd
import xgboost as xgb
import optuna
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config_loader import load_config, resolve_path

# Load config for config-driven paths
cfg = load_config("heart_disease")
processed_path = resolve_path(cfg, "data", "processed_path")
target_col = cfg["disease"]["target_column"]

def objective(trial):
    # تحميل البيانات
    data_path = os.path.join(processed_path, cfg["data"]["clinical_file"])
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # تحديد نطاق البحث الذكي للمعاملات
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    clf = xgb.XGBClassifier(**param, random_state=42, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)

# تشغيل البحث
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"🎯 أفضل دقة تم الوصول إليها: {study.best_value:.4f}")
print(f"⚙️ أفضل إعدادات: {study.best_params}")