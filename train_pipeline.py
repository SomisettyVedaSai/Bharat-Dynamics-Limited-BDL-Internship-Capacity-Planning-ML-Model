import pandas as pd
import numpy as np
import os
import pickle
import warnings
import json
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    data_path = os.path.join('data', 'bdl_production_planning_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data_generation.py first.")
    return pd.read_csv(data_path)

def preprocess_and_feature_engineer(df, is_train=True, preprocessors=None):
    df_copy = df.copy()
    
    # Custom Feature Engineering
    df_copy['effective_demand'] = df_copy['planning_period_qty'] / (1.0 - df_copy['rework_rate_pct'])
    df_copy['urgency_score'] = 1.0 / (df_copy['contracted_delivery_days'] + 1.0)
    df_copy['oee_age_factor'] = df_copy['machine_oee_pct'] * (1.0 - 0.005 * df_copy['machine_age_years'])
    df_copy['supply_risk_score'] = df_copy['vendor_lead_time_days'] * (1.0 - df_copy['indigenisation_pct'] / 100.0)
    df_copy['routing_criticality'] = df_copy['work_center_code'].apply(
        lambda x: 1 if x in ['WC_SEEKER', 'WC_INTEGRATION', 'WC_PROOF'] else 0
    )
    df_copy['load_per_line'] = df_copy['required_capacity_hrs'] / df_copy['num_parallel_lines']
    
    categorical_cols = [
        'weapon_system', 'work_center_code', 'sub_assembly_stage', 
        'manufacturing_unit', 'process_sheet_type', 'pgl_no', 
        'cost_center', 'shift_pattern', 'planning_period'
    ]
    
    numerical_cols = [
        'operation_time_min', 'setup_time_min', 'operation_sequence', 'total_smh',
        'contract_order_qty', 'planning_period_qty', 'contracted_delivery_days',
        'delivery_urgency_score', 'available_machine_hrs_day', 'num_parallel_lines',
        'working_days_in_period',
        'machine_oee_pct', 'machine_age_years', 'planned_downtime_hrs_month',
        'skilled_tech_available', 'fai_required_flag', 'qa_gate_clearance_hrs',
        'rework_rate_pct', 'drdo_signoff_required', 'vendor_lead_time_days',
        'indigenisation_pct', 'export_order_flag', 'effective_demand',
        'urgency_score', 'oee_age_factor', 'supply_risk_score', 'routing_criticality',
        'process_sheet_prep_days', 'drdo_signoff_lead_days', 'jig_fixture_prep_days', 'fai_setup_lead_days',
        'required_capacity_hrs', 'load_per_line'
    ]
    
    if is_train:
        # Fit Label Encoders
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoders[col] = le
            
        # Clip numerical outliers at 1st and 99th percentiles
        clip_thresholds = {}
        for col in numerical_cols:
            q01 = df_copy[col].quantile(0.01)
            q99 = df_copy[col].quantile(0.99)
            df_copy[col] = df_copy[col].clip(q01, q99)
            clip_thresholds[col] = (q01, q99)
            
        preprocessors = {
            'encoders': encoders,
            'clip_thresholds': clip_thresholds,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols
        }
    else:
        # Apply pre-fitted encoders and clipping thresholds
        encoders = preprocessors['encoders']
        clip_thresholds = preprocessors['clip_thresholds']
        
        for col in categorical_cols:
            le = encoders[col]
            df_copy[col] = df_copy[col].astype(str).map(
                lambda s: le.transform([s])[0] if s in le.classes_ else le.transform([le.classes_[0]])[0]
            )
            
        for col in numerical_cols:
            q01, q99 = clip_thresholds[col]
            df_copy[col] = df_copy[col].clip(q01, q99)
            
    return df_copy, preprocessors

def tune_model_A(X, y):
    print("Tuning Model A (XGBoost Regressor for required_capacity_hrs)...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': -1,
            'random_state': 42
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            scores.append(mean_squared_error(y_va, preds))
        return np.mean(scores)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=60, timeout=300)
    print(f"Model A Best Trial Score (MSE): {study.best_value:.4f}")
    return study.best_params

def tune_model_B(X, y):
    print("Tuning Model B (Random Forest Regressor for utilization_rate)...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'n_jobs': -1,
            'random_state': 42
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            model = RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            scores.append(mean_squared_error(y_va, preds))
        return np.mean(scores)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, timeout=300)
    print(f"Model B Best Trial Score (MSE): {study.best_value:.4f}")
    return study.best_params

def tune_model_C(X, y):
    print("Tuning Model C (XGBoost Classifier for bottleneck_flag)...")
    neg_samples = (y == 0).sum()
    pos_samples = (y == 1).sum()
    scale_weight = neg_samples / max(1, pos_samples)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': scale_weight,
            'n_jobs': -1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            scores.append(f1_score(y_va, preds, average='macro'))
        return np.mean(scores)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=60, timeout=300)
    print(f"Model C Best Trial Score (F1-macro): {study.best_value:.4f}")
    best_params = study.best_params.copy()
    best_params['scale_pos_weight'] = scale_weight
    return best_params

def tune_model_D(X, y):
    print("Tuning Model D (LightGBM Classifier for delivery_risk_flag)...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': -1
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            scores.append(f1_score(y_va, preds, average='macro'))
        return np.mean(scores)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=60, timeout=300)
    print(f"Model D Best Trial Score (F1-macro): {study.best_value:.4f}")
    return study.best_params

def tune_model_E(X, y):
    print("Tuning Model E (LightGBM Multi-class for overload_severity)...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': -1
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            scores.append(f1_score(y_va, preds, average='macro'))
        return np.mean(scores)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=60, timeout=300)
    print(f"Model E Best Trial Score (F1-macro): {study.best_value:.4f}")
    
    # Fix Bug 2: Remove objective and num_class from Optuna parameters returned to prevent LGBM constructor error
    best_params = study.best_params.copy()
    best_params.pop('objective', None)
    best_params.pop('num_class', None)
    return best_params

def get_oof_predictions_A(X_train_A, y_train_A, best_params_A):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train_A))
    for train_idx, val_idx in kf.split(X_train_A):
        X_tr, y_tr = X_train_A.iloc[train_idx], y_train_A.iloc[train_idx]
        X_va = X_train_A.iloc[val_idx]
        model = xgb.XGBRegressor(**best_params_A, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_va)
    return oof_preds

def main():
    # Load dataset
    df = load_data()
    
    # Prepare combined stratification column: weapon_system + bottleneck_flag
    df['stratify_col'] = df['weapon_system'].astype(str) + '_' + df['bottleneck_flag'].astype(str)
    
    # Train/Val/Test Split (70/15/15)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['stratify_col'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['stratify_col'])
    
    # Preprocess splits
    train_processed, preprocessors = preprocess_and_feature_engineer(train_df, is_train=True)
    val_processed, _ = preprocess_and_feature_engineer(val_df, is_train=False, preprocessors=preprocessors)
    test_processed, _ = preprocess_and_feature_engineer(test_df, is_train=False, preprocessors=preprocessors)
    
    # Define Feature Names
    features_A = preprocessors['categorical_cols'] + [
        c for c in preprocessors['numerical_cols'] if c not in ['load_per_line', 'required_capacity_hrs']
    ]
    features_other = features_A + ['required_capacity_hrs', 'load_per_line']
    
    print(f"Feature set A size: {len(features_A)}")
    print(f"Feature set other size: {len(features_other)}")
    
    # --- MODEL A: required_capacity_hrs ---
    X_train_A = train_processed[features_A]
    y_train_A = train_processed['required_capacity_hrs']
    X_val_A = val_processed[features_A]
    y_val_A = val_processed['required_capacity_hrs']
    X_test_A = test_processed[features_A]
    y_test_A = test_processed['required_capacity_hrs']
    
    best_params_A = tune_model_A(X_train_A, y_train_A)
    model_A = xgb.XGBRegressor(**best_params_A, random_state=42, n_jobs=-1)
    model_A.fit(X_train_A, y_train_A)
    
    preds_test_A = model_A.predict(X_test_A)
    
    # Generate Out-Of-Fold (OOF) predictions for Model A on the training set to eliminate train-predict gap
    print("Generating Out-of-Fold predictions of Model A on the training set...")
    oof_preds_A = get_oof_predictions_A(X_train_A, y_train_A, best_params_A)
    
    # --- PREPARE DATA FOR OTHER MODELS (using predicted/OOF required_capacity_hrs to prevent train-predict gap & leak) ---
    train_processed_pred = train_processed.copy()
    oof_series = pd.Series(oof_preds_A, index=X_train_A.index)
    train_processed_pred['required_capacity_hrs'] = oof_series
    train_processed_pred['load_per_line'] = oof_series / train_processed_pred['num_parallel_lines']
    
    X_train_other = train_processed_pred[features_other]
    y_train_B = train_processed['utilization_rate']
    y_train_C = train_processed['bottleneck_flag']
    y_train_D = train_processed['delivery_risk_flag']
    
    # Map overload severity labels
    severity_map = {'OK': 0, 'Warning': 1, 'Critical': 2}
    y_train_E = train_processed['overload_severity'].map(severity_map)
    
    # Build validation and test sets using Model A predictions
    val_preds_A = model_A.predict(X_val_A)
    val_processed_pred = val_processed.copy()
    val_processed_pred['required_capacity_hrs'] = val_preds_A
    val_processed_pred['load_per_line'] = val_preds_A / val_processed_pred['num_parallel_lines']
    X_val_other = val_processed_pred[features_other]
    
    test_preds_A = model_A.predict(X_test_A)
    test_processed_pred = test_processed.copy()
    test_processed_pred['required_capacity_hrs'] = test_preds_A
    test_processed_pred['load_per_line'] = test_preds_A / test_processed_pred['num_parallel_lines']
    X_test_other = test_processed_pred[features_other]
    
    y_test_B = test_processed['utilization_rate']
    y_test_C = test_processed['bottleneck_flag']
    y_test_D = test_processed['delivery_risk_flag']
    y_test_E = test_processed['overload_severity'].map(severity_map)
    
    # --- MODEL B: utilization_rate ---
    best_params_B = tune_model_B(X_train_A, y_train_B)
    model_B = RandomForestRegressor(**best_params_B, random_state=42, n_jobs=-1)
    model_B.fit(X_train_A, y_train_B)
    preds_test_B = model_B.predict(X_test_A)
    
    # --- MODEL C: bottleneck_flag ---
    best_params_C = tune_model_C(X_train_A, y_train_C)
    model_C = xgb.XGBClassifier(**best_params_C, random_state=42, n_jobs=-1)
    model_C.fit(X_train_A, y_train_C)
    preds_test_C = model_C.predict(X_test_A)
    probs_test_C = model_C.predict_proba(X_test_A)[:, 1]
    
    # --- MODEL D: delivery_risk_flag ---
    best_params_D = tune_model_D(X_train_other, y_train_D)
    model_D = lgb.LGBMClassifier(**best_params_D, random_state=42, n_jobs=-1, verbosity=-1)
    model_D.fit(X_train_other, y_train_D)
    preds_test_D = model_D.predict(X_test_other)
    probs_test_D = model_D.predict_proba(X_test_other)[:, 1]
    
    # --- MODEL E: overload_severity ---
    best_params_E = tune_model_E(X_train_A, y_train_E)
    model_E = lgb.LGBMClassifier(
        **best_params_E, 
        objective='multiclass', 
        num_class=3, 
        class_weight='balanced',
        random_state=42, 
        n_jobs=-1, 
        verbosity=-1
    )
    model_E.fit(X_train_A, y_train_E)
    preds_test_E = model_E.predict(X_test_A)

    # --- COMPREHENSIVE EVALUATION AND WRITING REPORTS ---
    performance_records = []
    evaluation_report_lines = []
    
    def log_and_record_line(line):
        print(line)
        evaluation_report_lines.append(line)
        
    log_and_record_line("=== CAPACITY PLANNING ENGINE EVALUATION REPORT ===")
    
    # 1. Evaluate Model A (required_capacity_hrs)
    overall_r2_A = r2_score(y_test_A, preds_test_A)
    overall_rmse_A = np.sqrt(mean_squared_error(y_test_A, preds_test_A))
    overall_mae_A = mean_absolute_error(y_test_A, preds_test_A)
    
    log_and_record_line("\n=== MODEL A EVALUATION (required_capacity_hrs) ===")
    log_and_record_line(f"Overall R²: {overall_r2_A:.4f}")
    log_and_record_line(f"Overall RMSE: {overall_rmse_A:.4f}")
    log_and_record_line(f"Overall MAE: {overall_mae_A:.4f}")
    
    performance_records.append({
        'Model': 'Model A (required_capacity_hrs)',
        'Weapon_System': 'OVERALL',
        'Metric_1': 'R2', 'Value_1': overall_r2_A,
        'Metric_2': 'RMSE', 'Value_2': overall_rmse_A,
        'Metric_3': 'MAE', 'Value_3': overall_mae_A
    })
    
    le_weapon = preprocessors['encoders']['weapon_system']
    log_and_record_line("\nModel A Performance by Weapon System:")
    per_weapon_A = {}
    for val in range(len(le_weapon.classes_)):
        weapon_name = le_weapon.classes_[val]
        mask = X_test_A['weapon_system'] == val
        if mask.sum() > 0:
            w_r2 = r2_score(y_test_A[mask], preds_test_A[mask])
            w_rmse = np.sqrt(mean_squared_error(y_test_A[mask], preds_test_A[mask]))
            w_mae = mean_absolute_error(y_test_A[mask], preds_test_A[mask])
            log_and_record_line(f"  {weapon_name:10s} -> R²: {w_r2:6.4f} | RMSE: {w_rmse:7.2f} hrs | MAE: {w_mae:7.2f} hrs")
            
            per_weapon_A[weapon_name] = {
                'r2': float(w_r2),
                'rmse': float(w_rmse),
                'mae': float(w_mae)
            }
            performance_records.append({
                'Model': 'Model A (required_capacity_hrs)',
                'Weapon_System': weapon_name,
                'Metric_1': 'R2', 'Value_1': w_r2,
                'Metric_2': 'RMSE', 'Value_2': w_rmse,
                'Metric_3': 'MAE', 'Value_3': w_mae
            })
            
    # 2. Evaluate Model B (utilization_rate)
    overall_r2_B = r2_score(y_test_B, preds_test_B)
    overall_rmse_B = np.sqrt(mean_squared_error(y_test_B, preds_test_B))
    overall_mae_B = mean_absolute_error(y_test_B, preds_test_B)
    
    log_and_record_line("\n=== MODEL B EVALUATION (utilization_rate) ===")
    log_and_record_line(f"Overall R²: {overall_r2_B:.4f}")
    log_and_record_line(f"Overall RMSE: {overall_rmse_B:.4f}")
    log_and_record_line(f"Overall MAE: {overall_mae_B:.4f}")
    
    performance_records.append({
        'Model': 'Model B (utilization_rate)',
        'Weapon_System': 'OVERALL',
        'Metric_1': 'R2', 'Value_1': overall_r2_B,
        'Metric_2': 'RMSE', 'Value_2': overall_rmse_B,
        'Metric_3': 'MAE', 'Value_3': overall_mae_B
    })
    
    log_and_record_line("\nModel B Performance by Weapon System:")
    for val in range(len(le_weapon.classes_)):
        weapon_name = le_weapon.classes_[val]
        mask = X_test_other['weapon_system'] == val
        if mask.sum() > 0:
            w_r2 = r2_score(y_test_B[mask], preds_test_B[mask])
            w_rmse = np.sqrt(mean_squared_error(y_test_B[mask], preds_test_B[mask]))
            w_mae = mean_absolute_error(y_test_B[mask], preds_test_B[mask])
            log_and_record_line(f"  {weapon_name:10s} -> R²: {w_r2:6.4f} | RMSE: {w_rmse:7.4f} | MAE: {w_mae:7.4f}")
            
            performance_records.append({
                'Model': 'Model B (utilization_rate)',
                'Weapon_System': weapon_name,
                'Metric_1': 'R2', 'Value_1': w_r2,
                'Metric_2': 'RMSE', 'Value_2': w_rmse,
                'Metric_3': 'MAE', 'Value_3': w_mae
            })

    # 3. Evaluate Model C (bottleneck_flag)
    overall_f1_C = f1_score(y_test_C, preds_test_C, average='macro')
    overall_rec_C = recall_score(y_test_C, preds_test_C, zero_division=0)
    overall_prec_C = precision_score(y_test_C, preds_test_C, zero_division=0)
    
    log_and_record_line("\n=== MODEL C EVALUATION (bottleneck_flag) ===")
    log_and_record_line(f"Overall F1-macro: {overall_f1_C:.4f}")
    log_and_record_line(f"Overall Recall (Bottlenecks): {overall_rec_C:.4f}")
    log_and_record_line(f"Overall Precision (Bottlenecks): {overall_prec_C:.4f}")
    
    performance_records.append({
        'Model': 'Model C (bottleneck_flag)',
        'Weapon_System': 'OVERALL',
        'Metric_1': 'F1_macro', 'Value_1': overall_f1_C,
        'Metric_2': 'Recall_class1', 'Value_2': overall_rec_C,
        'Metric_3': 'Precision_class1', 'Value_3': overall_prec_C
    })
    
    # Enrichment 6: Save Model C per-weapon system performance metrics
    log_and_record_line("\nModel C Performance by Weapon System:")
    per_weapon_C = {}
    for val in range(len(le_weapon.classes_)):
        weapon_name = le_weapon.classes_[val]
        mask = X_test_other['weapon_system'] == val
        if mask.sum() > 0:
            sub_y = y_test_C[mask]
            sub_pred = preds_test_C[mask]
            w_f1 = f1_score(sub_y, sub_pred, average='macro', zero_division=0)
            w_rec = recall_score(sub_y, sub_pred, zero_division=0)
            w_prec = precision_score(sub_y, sub_pred, zero_division=0)
            log_and_record_line(f"  {weapon_name:10s} -> F1-macro: {w_f1:6.4f} | Recall: {w_rec:6.4f} | Precision: {w_prec:6.4f}")
            
            per_weapon_C[weapon_name] = {
                'f1_macro': float(w_f1),
                'recall': float(w_rec),
                'precision': float(w_prec)
            }
            performance_records.append({
                'Model': 'Model C (bottleneck_flag)',
                'Weapon_System': weapon_name,
                'Metric_1': 'F1_macro', 'Value_1': w_f1,
                'Metric_2': 'Recall_class1', 'Value_2': w_rec,
                'Metric_3': 'Precision_class1', 'Value_3': w_prec
            })

    # 4. Evaluate Model D (delivery_risk_flag)
    overall_f1_D = f1_score(y_test_D, preds_test_D, average='macro')
    overall_rec_D = recall_score(y_test_D, preds_test_D, zero_division=0)
    overall_prec_D = precision_score(y_test_D, preds_test_D, zero_division=0)
    
    log_and_record_line("\n=== MODEL D EVALUATION (delivery_risk_flag) ===")
    log_and_record_line(f"Overall F1-macro: {overall_f1_D:.4f}")
    log_and_record_line(f"Overall Recall (Risk): {overall_rec_D:.4f}")
    log_and_record_line(f"Overall Precision (Risk): {overall_prec_D:.4f}")
    
    performance_records.append({
        'Model': 'Model D (delivery_risk_flag)',
        'Weapon_System': 'OVERALL',
        'Metric_1': 'F1_macro', 'Value_1': overall_f1_D,
        'Metric_2': 'Recall_class1', 'Value_2': overall_rec_D,
        'Metric_3': 'Precision_class1', 'Value_3': overall_prec_D
    })
    
    log_and_record_line("\nModel D Performance by Weapon System:")
    for val in range(len(le_weapon.classes_)):
        weapon_name = le_weapon.classes_[val]
        mask = X_test_other['weapon_system'] == val
        if mask.sum() > 0:
            sub_y = y_test_D[mask]
            sub_pred = preds_test_D[mask]
            w_f1 = f1_score(sub_y, sub_pred, average='macro', zero_division=0)
            w_rec = recall_score(sub_y, sub_pred, zero_division=0)
            w_prec = precision_score(sub_y, sub_pred, zero_division=0)
            log_and_record_line(f"  {weapon_name:10s} -> F1-macro: {w_f1:6.4f} | Recall: {w_rec:6.4f} | Precision: {w_prec:6.4f}")
            
            performance_records.append({
                'Model': 'Model D (delivery_risk_flag)',
                'Weapon_System': weapon_name,
                'Metric_1': 'F1_macro', 'Value_1': w_f1,
                'Metric_2': 'Recall_class1', 'Value_2': w_rec,
                'Metric_3': 'Precision_class1', 'Value_3': w_prec
            })

    # 5. Evaluate Model E (overload_severity)
    overall_f1_E = f1_score(y_test_E, preds_test_E, average='macro')
    overall_acc_E = (preds_test_E == y_test_E).mean()
    
    log_and_record_line("\n=== MODEL E EVALUATION (overload_severity) ===")
    log_and_record_line(f"Overall F1-macro: {overall_f1_E:.4f}")
    log_and_record_line(f"Overall Accuracy: {overall_acc_E:.4f}")
    
    performance_records.append({
        'Model': 'Model E (overload_severity)',
        'Weapon_System': 'OVERALL',
        'Metric_1': 'F1_macro', 'Value_1': overall_f1_E,
        'Metric_2': 'Accuracy', 'Value_2': overall_acc_E,
        'Metric_3': 'N/A', 'Value_3': 0.0
    })
    
    log_and_record_line("\nModel E Performance by Weapon System:")
    for val in range(len(le_weapon.classes_)):
        weapon_name = le_weapon.classes_[val]
        mask = X_test_other['weapon_system'] == val
        if mask.sum() > 0:
            sub_y = y_test_E[mask]
            sub_pred = preds_test_E[mask]
            w_f1 = f1_score(sub_y, sub_pred, average='macro', zero_division=0)
            w_acc = (sub_pred == sub_y).mean()
            log_and_record_line(f"  {weapon_name:10s} -> F1-macro: {w_f1:6.4f} | Accuracy: {w_acc:6.4f}")
            
            performance_records.append({
                'Model': 'Model E (overload_severity)',
                'Weapon_System': weapon_name,
                'Metric_1': 'F1_macro', 'Value_1': w_f1,
                'Metric_2': 'Accuracy', 'Value_2': w_acc,
                'Metric_3': 'N/A', 'Value_3': 0.0
            })

    # Save performance records as CSV
    os.makedirs('models', exist_ok=True)
    perf_df = pd.DataFrame(performance_records)
    perf_df.to_csv('models/model_performance_metrics.csv', index=False)
    print("Saved models/model_performance_metrics.csv")
    
    # Save evaluation report as TXT
    with open('models/evaluation_report.txt', 'w') as f:
        f.write('\n'.join(evaluation_report_lines))
    print("Saved models/evaluation_report.txt")
    
    # Save Model C & D AUCs and class reports for JSON report
    auc_C = float(roc_auc_score(y_test_C, probs_test_C))
    auc_D = float(roc_auc_score(y_test_D, probs_test_D))
    cr_C = classification_report(y_test_C, preds_test_C, output_dict=True)
    cr_D = classification_report(y_test_D, preds_test_D, output_dict=True)
    cr_E = classification_report(y_test_E, preds_test_E, output_dict=True)
    
    # Enrichment 6: Save evaluation report JSON
    report = {
        'model_A': {
            'r2': float(overall_r2_A),
            'rmse': float(overall_rmse_A),
            'mae': float(overall_mae_A),
            'per_weapon': per_weapon_A
        },
        'model_B': {
            'r2': float(overall_r2_B),
            'rmse': float(overall_rmse_B),
            'mae': float(overall_mae_B)
        },
        'model_C': {
            'classification_report': cr_C,
            'roc_auc': auc_C,
            'per_weapon': per_weapon_C
        },
        'model_D': {
            'classification_report': cr_D,
            'roc_auc': auc_D
        },
        'model_E': {
            'classification_report': cr_E
        }
    }
    
    with open('models/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("Saved models/evaluation_report.json")

    # Save models and preprocessors
    with open('models/preprocessors.pkl', 'wb') as f:
        pickle.dump(preprocessors, f)
    with open('models/model_A.pkl', 'wb') as f:
        pickle.dump(model_A, f)
    with open('models/model_B.pkl', 'wb') as f:
        pickle.dump(model_B, f)
    with open('models/model_C.pkl', 'wb') as f:
        pickle.dump(model_C, f)
    with open('models/model_D.pkl', 'wb') as f:
        pickle.dump(model_D, f)
    with open('models/model_E.pkl', 'wb') as f:
        pickle.dump(model_E, f)
    
    print("\nAll models and preprocessors saved to models/")

if __name__ == '__main__':
    main()
