import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("Starting Exploratory Data Analysis...")
    
    # Load dataset
    data_path = os.path.join('data', 'bdl_production_planning_data.csv')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run data_generation.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Style settings
    sns.set_theme(style="darkgrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # 1. Distribution: operation_time_min and total_smh by weapon_system
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Sort weapon systems for consistent plotting
    weapon_order = sorted(df['weapon_system'].unique())
    
    sns.boxplot(ax=axes[0], data=df, x='weapon_system', y='operation_time_min', order=weapon_order, palette='Set2')
    axes[0].set_title('Operation Time (minutes) Distribution by Weapon System', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Weapon System', fontsize=12)
    axes[0].set_ylabel('Operation Time (min)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(ax=axes[1], data=df, x='weapon_system', y='total_smh', order=weapon_order, palette='Set2')
    axes[1].set_title('Total Standard Minute Hours (SMH) by Weapon System', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Weapon System', fontsize=12)
    axes[1].set_ylabel('Total SMH', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/operation_and_smh_distributions.png', dpi=300)
    plt.close()
    print("1. Saved operation_and_smh_distributions.png")
    
    # 2. Heatmap: utilization_rate by work_center_code vs weapon_system
    plt.figsize = (12, 8)
    pivot_df = df.pivot_table(values='utilization_rate', index='work_center_code', columns='weapon_system', aggfunc='mean')
    
    # Sort work centers by sequence order for a logical flow
    wc_sequence = ['WC_MACH', 'WC_SHEET', 'WC_SMT', 'WC_ELEC', 'WC_SEEKER', 'WC_PROP', 'WC_WARHEAD', 
                   'WC_TORPEDO', 'WC_INTEGRATION', 'WC_LAUNCHER', 'WC_PROOF', 'WC_QA_INSP', 'WC_PACK']
    # Filter only WCs present in data
    wc_sequence = [wc for wc in wc_sequence if wc in pivot_df.index]
    pivot_df = pivot_df.reindex(wc_sequence)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Average Utilization Rate'})
    plt.title('Average Work Center Utilization Rate by Weapon System', fontsize=16, fontweight='bold')
    plt.ylabel('Work Center Code', fontsize=12)
    plt.xlabel('Weapon System', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/utilization_heatmap.png', dpi=300)
    plt.close()
    print("2. Saved utilization_heatmap.png")
    
    # 3. Bar chart: average bottleneck rate per work_center
    plt.figure(figsize=(12, 6))
    wc_bottlenecks = df.groupby('work_center_code')['bottleneck_flag'].mean().reset_index()
    # Sort by sequence
    wc_bottlenecks['seq'] = wc_bottlenecks['work_center_code'].map(lambda x: wc_sequence.index(x) if x in wc_sequence else 99)
    wc_bottlenecks = wc_bottlenecks.sort_values('seq')
    
    sns.barplot(data=wc_bottlenecks, x='work_center_code', y='bottleneck_flag', palette='Reds_d')
    plt.title('Average Bottleneck Rate per Work Center', fontsize=16, fontweight='bold')
    plt.xlabel('Work Center Code', fontsize=12)
    plt.ylabel('Bottleneck Probability (utilization > 0.85 & critical WC)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/bottleneck_rates.png', dpi=300)
    plt.close()
    print("3. Saved bottleneck_rates.png")
    
    # 4. Time series: capacity load over contracted_delivery_days (urgency curve)
    plt.figure(figsize=(12, 6))
    # Bin contracted delivery days into 30-day windows and calculate mean utilization
    df['delivery_days_bin'] = (df['contracted_delivery_days'] // 30) * 30
    urgency_df = df.groupby('delivery_days_bin')['utilization_rate'].mean().reset_index()
    
    sns.lineplot(data=urgency_df, x='delivery_days_bin', y='utilization_rate', marker='o', color='crimson', linewidth=2.5)
    plt.axhline(y=0.85, color='orange', linestyle='--', label='Warning Threshold (85%)')
    plt.axhline(y=1.0, color='red', linestyle='--', label='Overload Threshold (100%)')
    plt.title('Average Capacity Load (Utilization) vs Days to Delivery (Urgency Curve)', fontsize=16, fontweight='bold')
    plt.xlabel('Contracted Days to Delivery (30-day Bins)', fontsize=12)
    plt.ylabel('Average Utilization Rate', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/urgency_curve.png', dpi=300)
    plt.close()
    print("4. Saved urgency_curve.png")
    
    # 5. Box plot: rework_rate by process_sheet_type and sub_assembly_stage
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    sns.boxplot(ax=axes[0], data=df, x='process_sheet_type', y='rework_rate_pct', palette='Pastel1')
    axes[0].set_title('Rework Rate (%) by Process Sheet Type', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Process Sheet Type', fontsize=12)
    axes[0].set_ylabel('Rework Rate (%)', fontsize=12)
    
    sns.boxplot(ax=axes[1], data=df, x='sub_assembly_stage', y='rework_rate_pct', palette='Pastel2')
    axes[1].set_title('Rework Rate (%) by Sub-Assembly Stage', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Sub-Assembly Stage', fontsize=12)
    axes[1].set_ylabel('Rework Rate (%)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.savefig('plots/rework_rate_by_type_stage.png', dpi=300)
    plt.close()
    print("5. Saved rework_rate_by_type_stage.png")
    print("EDA Complete!")

if __name__ == '__main__':
    run_eda()
