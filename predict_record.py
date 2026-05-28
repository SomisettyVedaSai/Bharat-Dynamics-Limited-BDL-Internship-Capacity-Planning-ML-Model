import json
from inference import predict_capacity

def run_interactive_demo():
    print("==================================================")
    print("| BHARAT DYNAMICS LIMITED - CAPACITY INFERENCE DEMO |")
    print("==================================================")
    print("Enter the parameters for your planning record:\n")
    
    # 1. Select Weapon System
    weapons = ['Nag', 'Akash', 'Milan2T', 'Konkurs', 'Astra', 'HWT', 'LWT', 'Amogha3', 'HELINA', 'CMDS']
    print("Select Weapon System:")
    for idx, w in enumerate(weapons):
        print(f"  [{idx}] {w}")
    try:
        w_choice = int(input("Choice (0-9, default 0): ") or 0)
        weapon = weapons[w_choice]
    except Exception:
        weapon = 'Nag'
        
    # 2. Select Work Center
    work_centers = ['WC_SEEKER', 'WC_INTEGRATION', 'WC_PROOF', 'WC_MACH', 'WC_SHEET', 'WC_SMT', 'WC_ELEC', 'WC_PROP', 'WC_WARHEAD', 'WC_LAUNCHER', 'WC_TORPEDO', 'WC_QA_INSP', 'WC_PACK']
    print("\nSelect Work Center:")
    for idx, wc in enumerate(work_centers):
        print(f"  [{idx}] {wc}")
    try:
        wc_choice = int(input("Choice (0-12, default 0): ") or 0)
        work_center = work_centers[wc_choice]
    except Exception:
        work_center = 'WC_SEEKER'
        
    # 3. Planning period quantity
    try:
        qty = int(input("\nEnter Planning Period Quantity (2-100, default 15): ") or 15)
    except Exception:
        qty = 15
        
    # 4. Shifts
    shifts = ['Single', 'Double', 'Triple']
    print("\nSelect Shift Pattern:")
    for idx, s in enumerate(shifts):
        print(f"  [{idx}] {s}")
    try:
        s_choice = int(input("Choice (0-2, default 0): ") or 0)
        shift = shifts[s_choice]
    except Exception:
        shift = 'Single'
        
    # 5. Parallel lines
    try:
        lines = int(input("\nEnter Number of Parallel Lines (1-10, default 1): ") or 1)
    except Exception:
        lines = 1
        
    # 6. OEE
    try:
        oee = float(input("\nEnter Machine OEE (0.55-0.92, default 0.75): ") or 0.75)
    except Exception:
        oee = 0.75
        
    # Helper configurations based on BDL routing
    unit_mappings = {
        'Nag': { 'unit': 'Hyderabad', 'cc': 'CC_HYD', 'pgl': 'PGL_NAG_ATGM', 'drdo': 1, 'indigenisation': 85 },
        'Akash': { 'unit': 'Ibrahimpatnam', 'cc': 'CC_IBP', 'pgl': 'PGL_AKASH_SAM', 'drdo': 1, 'indigenisation': 90 },
        'Milan2T': { 'unit': 'Bhanur', 'cc': 'CC_BHANUR', 'pgl': 'PGL_MILAN_ATGM', 'drdo': 0, 'indigenisation': 60 },
        'Konkurs': { 'unit': 'Bhanur', 'cc': 'CC_BHANUR', 'pgl': 'PGL_KONKURS_ATGM', 'drdo': 0, 'indigenisation': 55 },
        'Astra': { 'unit': 'Hyderabad', 'cc': 'CC_HYD', 'pgl': 'PGL_ASTRA_AAM', 'drdo': 1, 'indigenisation': 80 },
        'HWT': { 'unit': 'Visakhapatnam', 'cc': 'CC_VIZAG', 'pgl': 'PGL_TORPEDO_HWT', 'drdo': 1, 'indigenisation': 75 },
        'LWT': { 'unit': 'Visakhapatnam', 'cc': 'CC_VIZAG', 'pgl': 'PGL_TORPEDO_LWT', 'drdo': 1, 'indigenisation': 70 },
        'Amogha3': { 'unit': 'Hyderabad', 'cc': 'CC_HYD', 'pgl': 'PGL_AMOGHA_ATGM', 'drdo': 1, 'indigenisation': 92 },
        'HELINA': { 'unit': 'Hyderabad', 'cc': 'CC_HYD', 'pgl': 'PGL_HELINA_ATGM', 'drdo': 1, 'indigenisation': 88 },
        'CMDS': { 'unit': 'Bhanur', 'cc': 'CC_BHANUR', 'pgl': 'PGL_CMDS', 'drdo': 0, 'indigenisation': 95 }
    }
    
    wc_mappings = {
        'WC_MACH': { 'stage': 'Mech_Fab', 'seq': 1, 'op_time': 45, 'setup': 45 },
        'WC_SHEET': { 'stage': 'Mech_Fab', 'seq': 2, 'op_time': 35, 'setup': 30 },
        'WC_SMT': { 'stage': 'Electronics_Assy', 'seq': 3, 'op_time': 120, 'setup': 60 },
        'WC_ELEC': { 'stage': 'Electronics_Assy', 'seq': 4, 'op_time': 180, 'setup': 45 },
        'WC_SEEKER': { 'stage': 'Seeker_Assy', 'seq': 5, 'op_time': 300, 'setup': 90 },
        'WC_PROP': { 'stage': 'Propulsion', 'seq': 6, 'op_time': 200, 'setup': 120 },
        'WC_WARHEAD': { 'stage': 'Warhead', 'seq': 7, 'op_time': 150, 'setup': 90 },
        'WC_TORPEDO': { 'stage': 'Integration', 'seq': 8, 'op_time': 600, 'setup': 180 },
        'WC_INTEGRATION': { 'stage': 'Integration', 'seq': 9, 'op_time': 360, 'setup': 120 },
        'WC_LAUNCHER': { 'stage': 'Integration', 'seq': 10, 'op_time': 400, 'setup': 90 },
        'WC_PROOF': { 'stage': 'Proof_Test', 'seq': 11, 'op_time': 240, 'setup': 60 },
        'WC_QA_INSP': { 'stage': 'QA_Inspect', 'seq': 12, 'op_time': 90, 'setup': 15 },
        'WC_PACK': { 'stage': 'QA_Inspect', 'seq': 13, 'op_time': 45, 'setup': 15 }
    }

    config = unit_mappings[weapon]
    wc_config = wc_mappings[work_center]
    
    # Calculate available hours per day per shift
    shift_mult = 1 if shift == 'Single' else (2 if shift == 'Double' else 3)
    available_hrs = float(8 * shift_mult)
    
    # Construct complete 30-feature record dictionary
    record = {
        # Original Process Sheet Parameters
        'work_center_code': work_center,
        'machine_code': f"{work_center}_LINE_1",
        'operation_time_min': float(wc_config['op_time']),
        'setup_time_min': float(wc_config['setup']),
        'process_sheet_type': 'Standard',
        'pgl_no': config['pgl'],
        'operation_sequence': int(wc_config['seq']),
        'total_smh': float((wc_config['op_time'] * qty) / 60.0),
        'cost_center': config['cc'],
        
        # BDL Defence-Specific Parameters
        'weapon_system': weapon,
        'sub_assembly_stage': wc_config['stage'],
        'manufacturing_unit': config['unit'],
        'contract_order_qty': int(qty + 200),
        'planning_period_qty': int(qty),
        'contracted_delivery_days': 60,
        'delivery_urgency_score': 1.0 / 61.0,
        
        # Capacity & Efficiency Parameters
        'available_machine_hrs_day': available_hrs,
        'shift_pattern': shift,
        'num_parallel_lines': int(lines),
        'machine_oee_pct': float(oee),
        'machine_age_years': 22 if config['unit'] == 'Hyderabad' else (12 if config['unit'] == 'Bhanur' else 5),
        'planned_downtime_hrs_month': 15.0,
        'skilled_tech_available': int(lines * shift_mult + 1),
        
        # Quality & Compliance Parameters
        'fai_required_flag': 0,
        'qa_gate_clearance_hrs': 4.0,
        'rework_rate_pct': 0.05,
        'drdo_signoff_required': config['drdo'],
        
        # Supply Chain Parameters
        'vendor_lead_time_days': 120,
        'indigenisation_pct': float(config['indigenisation']),
        'export_order_flag': 0,
        
        # Process Sheet lead times
        'process_sheet_prep_days': 10.0,
        'drdo_signoff_lead_days': float(20.0 if config['drdo'] == 1 else 0.0),
        'jig_fixture_prep_days': 15.0,
        'fai_setup_lead_days': 0.0
    }
    
    print("\nRunning ML Pipeline Prediction...")
    pred = predict_capacity(record)
    
    print("\n================ PIPELINE PREDICTION OUTPUT ================")
    print(f"Weapon System:          {weapon}")
    print(f"Work Center:            {work_center}")
    print(f"Required Capacity:      {pred['required_capacity_hrs']:.2f} Hours")
    print(f"Work Center Util Rate:  {pred['utilization_rate']*100:.2f}%")
    print(f"Bottleneck WC Flag:     {'[!!!] BOTTLENECK DETECTED' if pred['bottleneck_flag'] == 1 else 'No Bottleneck'}")
    print(f"Delivery Risk:          {'[!!!] HIGH DELIVERY RISK' if pred['delivery_risk_flag'] == 1 else 'On-Schedule'}")
    print(f"Overload Severity:      {pred['overload_severity']}")
    print("\nTop 3 SHAP Reasons for Bottleneck:")
    for reason in pred['shap_top3_reasons']:
        print(f"  - {reason}")
    print("============================================================\n")

if __name__ == '__main__':
    run_interactive_demo()
