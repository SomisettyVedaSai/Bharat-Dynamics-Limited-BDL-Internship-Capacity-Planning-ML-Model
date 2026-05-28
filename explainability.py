import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from train_pipeline import preprocess_and_feature_engineer

warnings.filterwarnings('ignore')

def main():
    print("Running SHAP Explainability analysis...")
    
    # Check paths
    if not os.path.exists('models/model_C.pkl') or not os.path.exists('models/preprocessors.pkl'):
        print("Error: Models/Preprocessors not found. Run train_pipeline.py first.")
        return
        
    # Load model and preprocessor
    with open('models/model_C.pkl', 'rb') as f:
        model_C = pickle.load(f)
    with open('models/model_B.pkl', 'rb') as f:
        model_B = pickle.load(f)
    with open('models/preprocessors.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
        
    df = pd.read_csv('data/bdl_production_planning_data.csv')
    
    # Sample a set of rows for SHAP background (e.g. 500 rows)
    df_processed, _ = preprocess_and_feature_engineer(df, is_train=False, preprocessors=preprocessors)
    
    # Feature names
    features_A = preprocessors['categorical_cols'] + [
        c for c in preprocessors['numerical_cols'] if c not in ['load_per_line', 'required_capacity_hrs']
    ]
    features_other = features_A + ['required_capacity_hrs', 'load_per_line']
    
    X = df_processed[features_A]
    
    # Create tree explainer for Model C (XGBoost Classifier)
    explainer_C = shap.TreeExplainer(model_C)
    
    # Get SHAP values for a subset to save time
    shap_sample = X.sample(500, random_state=42)
    shap_values_C = explainer_C(shap_sample)
    
    # 1. SHAP Beeswarm Plot (overall feature importance)
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_C, max_display=15, show=False)
    plt.title('SHAP Feature Importance (Beeswarm) - Bottleneck Flag Classifier', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/shap_beeswarm.png', dpi=300)
    plt.close()
    print("1. Saved shap_beeswarm.png")
    
    # 2. SHAP Force Plot for a specific Nag ATGM planning record
    nag_record = {
        'work_center_code': 'WC_SEEKER',
        'machine_code': 'SEEKER_TEST_01',
        'operation_time_min': 300.0,
        'setup_time_min': 60.0,
        'process_sheet_type': 'FAI',
        'pgl_no': 'PGL_NAG_ATGM',
        'operation_sequence': 5,
        'total_smh': 90.0, # (300 * 18) / 60
        'cost_center': 'CC_HYD',
        'weapon_system': 'Nag',
        'sub_assembly_stage': 'Seeker_Assy',
        'manufacturing_unit': 'Hyderabad',
        'contract_order_qty': 200,
        'planning_period_qty': 18,
        'contracted_delivery_days': 60,
        'delivery_urgency_score': 1.0 / 61.0,
        'available_machine_hrs_day': 8.0, # Single shift
        'shift_pattern': 'Single',
        'num_parallel_lines': 1,
        'working_days_in_period': 22,
        'machine_oee_pct': 0.61,
        'machine_age_years': 22,
        'planned_downtime_hrs_month': 15.0,
        'skilled_tech_available': 2,
        'fai_required_flag': 1,
        'qa_gate_clearance_hrs': 4.0,
        'rework_rate_pct': 0.05,
        'drdo_signoff_required': 1,
        'vendor_lead_time_days': 120,
        'indigenisation_pct': 85.0,
        'export_order_flag': 0,
        'process_sheet_prep_days': 25.0,
        'drdo_signoff_lead_days': 35.0,
        'jig_fixture_prep_days': 30.0,
        'fai_setup_lead_days': 20.0,
        'planning_period': '2023-03',
        # Targets (which will be computed or predicted)
        'required_capacity_hrs': 155.0, # roughly calculated
        'utilization_rate': 155.0 / (8.0 * 25 * 1), # 155 / 200 = 0.775
        'bottleneck_flag': 1,
        'delivery_risk_flag': 0,
        'overload_severity': 'Warning'
    }
    
    # Fix Bug 7: Strip target columns from nag_record before converting to DataFrame and preprocessing
    TARGET_COLS = ['required_capacity_hrs', 'utilization_rate', 'bottleneck_flag',
                   'delivery_risk_flag', 'overload_severity']
    nag_df = pd.DataFrame([{k: v for k, v in nag_record.items() if k not in TARGET_COLS}])
    nag_df['required_capacity_hrs'] = 0.0 # Add dummy placeholder to prevent KeyError in preprocessor
    nag_processed, _ = preprocess_and_feature_engineer(nag_df, is_train=False, preprocessors=preprocessors)
    
    # Update capacity and load per line based on Model A's actual calculation or simple prediction
    nag_processed['required_capacity_hrs'] = 207.6
    nag_processed['load_per_line'] = 207.6 / 1.0
    
    X_nag = nag_processed[features_A]
    
    # Calculate SHAP values for this specific record
    shap_nag = explainer_C(X_nag)
    
    # Draw SHAP force plot for this record and save as image
    plt.figure(figsize=(12, 4))
    shap.plots.force(shap_nag[0], matplotlib=True, show=False)
    plt.title('SHAP Force Plot: Explanation for Nag ATGM Bottleneck at WC_SEEKER', fontsize=14, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig('plots/shap_force_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("2. Saved shap_force_plot.png")
    
    # 3. SHAP Dependence Plot: utilization_rate vs machine_oee_pct coloured by weapon_system
    explainer_B = shap.TreeExplainer(model_B)
    shap_values_B = explainer_B(shap_sample)
    
    # Plot dependence
    plt.figure(figsize=(10, 6))
    oee_idx = features_A.index('machine_oee_pct')
    weapon_idx = features_A.index('weapon_system')
    
    # Generate scatter plot manually to color by weapon system name
    oee_vals = shap_sample['machine_oee_pct']
    shap_oee = shap_values_B.values[:, oee_idx]
    weapon_vals = shap_sample['weapon_system']
    
    # Map back weapon system names
    weapon_names = preprocessors['encoders']['weapon_system'].inverse_transform(weapon_vals)
    
    sns.scatterplot(x=oee_vals, y=shap_oee, hue=weapon_names, palette='tab10', alpha=0.8)
    plt.title('SHAP Dependence: Utilization vs Machine OEE (colored by Weapon System)', fontsize=14, fontweight='bold')
    plt.xlabel('Machine OEE %', fontsize=12)
    plt.ylabel('SHAP Value for machine_oee_pct (contribution to utilization)', fontsize=12)
    plt.legend(title='Weapon System', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/shap_dependence.png', dpi=300)
    plt.close()
    print("3. Saved shap_dependence.png")
    
    # 4. Generate Text-based summary explanation
    contribs = []
    for j, col in enumerate(features_A):
        val = X_nag.iloc[0][col]
        shap_val = shap_nag.values[0, j]
        contribs.append((col, val, shap_val))
        
    # Sort by SHAP value descending (most positive contribution first)
    contribs = sorted(contribs, key=lambda x: x[2], reverse=True)
    
    # Format values back for readability
    reasons = []
    for col, val, s_val in contribs[:4]:
        if col == 'weapon_system':
            name = preprocessors['encoders']['weapon_system'].inverse_transform([int(val)])[0]
            reasons.append(f"weapon_system={name}")
        elif col == 'fai_required_flag':
            reasons.append(f"fai_required={'True' if val == 1 else 'False'}")
        elif col == 'machine_oee_pct':
            reasons.append(f"machine_oee={val:.2f}")
        elif col == 'num_parallel_lines':
            reasons.append(f"parallel_lines={int(val)}")
        elif col == 'planning_period_qty':
            reasons.append(f"planning_period_qty={int(val)}")
        else:
            reasons.append(f"{col}={val}")
            
    summary_str = f"WC_SEEKER bottlenecked because: {' + '.join(reasons)}"
    print(f"\nSummary text output:\n\"{summary_str}\"")
    
    with open('plots/shap_summary.txt', 'w') as f:
        f.write(summary_str)

if __name__ == '__main__':
    main()
