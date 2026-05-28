import os
import pandas as pd
import plotly.express as px
import plotly.io as pio

def generate_dashboard():
    print("Generating standalone Plotly Capacity Heatmap HTML...")
    
    # Load dataset
    data_path = os.path.join('data', 'bdl_production_planning_data.csv')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data_generation.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Sort work centers for logical sequence
    wc_seq = ['WC_MACH', 'WC_SHEET', 'WC_SMT', 'WC_ELEC', 'WC_SEEKER', 'WC_PROP', 'WC_WARHEAD', 
              'WC_TORPEDO', 'WC_INTEGRATION', 'WC_LAUNCHER', 'WC_PROOF', 'WC_QA_INSP', 'WC_PACK']
    
    pivot_df = df.pivot_table(values='utilization_rate', index='work_center_code', columns='weapon_system', aggfunc='mean').reset_index()
    # Filter and sort
    pivot_df['seq'] = pivot_df['work_center_code'].map(lambda x: wc_seq.index(x) if x in wc_seq else 99)
    pivot_df = pivot_df.sort_values('seq').drop(columns=['seq'])
    pivot_df = pivot_df.set_index('work_center_code')
    
    # Create Plotly Heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Weapon System", y="Work Center", color="Avg Capacity Utilization"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="RdYlGn_r", # Red for high utilization, green for low
        aspect="auto",
        title="Bharat Dynamics Limited - Capacity Utilization Heatmap"
    )
    fig.update_layout(
        font=dict(family="Outfit, sans-serif", size=14),
        title_font=dict(size=18, family="Outfit"),
        coloraxis_colorbar=dict(title="Utilization Rate", tickformat=".0%"),
        width=1000,
        height=700
    )
    
    # Save as standalone HTML
    out_path = 'capacity_heatmap_dashboard.html'
    pio.write_html(fig, out_path)
    print(f"Standalone Plotly dashboard saved to: {out_path}")

if __name__ == '__main__':
    generate_dashboard()
