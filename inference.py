import os
import pickle
import numpy as np
import pandas as pd
import shap
import warnings

warnings.filterwarnings('ignore')

# Global variables for models and preprocessors
models = {}
preprocessors = None
explainer_C = None
features_other = None
features_A = None

SHAP_TRANSLATION_MAP = {
    'work_center_code': 'Work Center Code',
    'machine_code': 'Machine Code',
    'operation_time_min': 'Operation Time (mins)',
    'setup_time_min': 'Setup Time (mins)',
    'process_sheet_type': 'Process Sheet Type',
    'pgl_no': 'PGL No.',
    'operation_sequence': 'Operation Sequence',
    'total_smh': 'Total SMH',
    'cost_center': 'Cost Center',
    'weapon_system': 'Weapon System',
    'sub_assembly_stage': 'Sub-assembly Stage',
    'manufacturing_unit': 'Manufacturing Unit',
    'contract_order_qty': 'Contract Order Qty',
    'planning_period_qty': 'Planning Period Qty',
    'contracted_delivery_days': 'Delivery SLA Days',
    'delivery_urgency_score': 'Delivery Urgency Score',
    'planning_period': 'Planning Period',
    'available_machine_hrs_day': 'Machine Available Hrs/Day',
    'shift_pattern': 'Shift Pattern',
    'num_parallel_lines': 'Parallel Lines Count',
    'machine_oee_pct': 'Machine OEE',
    'machine_age_years': 'Machine Age (years)',
    'planned_downtime_hrs_month': 'Planned Downtime (hrs/month)',
    'skilled_tech_available': 'Skilled Techs Available',
    'fai_required_flag': 'First Article Inspection (FAI)',
    'qa_gate_clearance_hrs': 'QA Gate Clearance (hrs)',
    'rework_rate_pct': 'Rework Rate %',
    'drdo_signoff_required': 'DRDO Sign-off Required',
    'vendor_lead_time_days': 'Vendor Lead Time (days)',
    'indigenisation_pct': 'Indigenisation %',
    'export_order_flag': 'Export Order Flag',
    'process_sheet_prep_days': 'Process Sheet Prep Days',
    'drdo_signoff_lead_days': 'DRDO Sign-off Lead Days',
    'jig_fixture_prep_days': 'Jig & Fixture Prep Days',
    'fai_setup_lead_days': 'FAI Setup Lead Days',
    'required_capacity_hrs': 'Required Capacity (hrs)',
    'load_per_line': 'Line Load (hrs)',
    'utilization_rate': 'Capacity Utilization',
    'effective_demand': 'Effective Demand',
    'urgency_score': 'Urgency Score',
    'oee_age_factor': 'OEE Age Factor',
    'supply_risk_score': 'Supply Chain Risk',
    'routing_criticality': 'Critical Path Stage',
    'working_days_in_period': 'Working Days in Period'
}

def init_models():
    global models, preprocessors, explainer_C, features_other, features_A
    if len(models) > 0:
        return
        
    model_dir = 'models'
    if not os.path.exists(os.path.join(model_dir, 'preprocessors.pkl')):
        raise FileNotFoundError("Models or preprocessors not found. Please run train_pipeline.py first.")
        
    with open(os.path.join(model_dir, 'preprocessors.pkl'), 'rb') as f:
        preprocessors = pickle.load(f)
        
    model_names = ['A', 'B', 'C', 'D', 'E']
    for m in model_names:
        with open(os.path.join(model_dir, f'model_{m}.pkl'), 'rb') as f:
            models[m] = pickle.load(f)
            
    # Initialize explainer for Model C
    try:
        explainer_C = shap.TreeExplainer(models['C'])
    except Exception as e:
        print(f"Failed to initialize SHAP TreeExplainer: {e}")
        explainer_C = None
    
    # Feature lists
    features_A = preprocessors['categorical_cols'] + [
        c for c in preprocessors['numerical_cols'] if c not in ['load_per_line', 'required_capacity_hrs']
    ]
    features_other = features_A + ['required_capacity_hrs', 'load_per_line']

def validate_record(record: dict) -> tuple[bool, str]:
    required_fields = [
        'work_center_code', 'weapon_system', 'operation_time_min',
        'setup_time_min', 'planning_period_qty', 'contracted_delivery_days',
        'machine_oee_pct', 'num_parallel_lines', 'shift_pattern',
        'available_machine_hrs_day', 'rework_rate_pct', 'fai_required_flag',
        'drdo_signoff_required', 'vendor_lead_time_days', 'indigenisation_pct',
        'working_days_in_period'
    ]
    for field in required_fields:
        if field not in record:
            return False, f"Missing required field: {field}"
    # Range checks
    if not (0.55 <= float(record['machine_oee_pct']) <= 0.95):
        return False, "machine_oee_pct must be between 0.55 and 0.95"
    if not (1 <= int(record['num_parallel_lines']) <= 20):
        return False, "num_parallel_lines must be between 1 and 20"
    if not (10 <= int(record['working_days_in_period']) <= 28):
        return False, "working_days_in_period must be between 10 and 28"
    if int(record['planning_period_qty']) < 1:
        return False, "planning_period_qty must be >= 1"
    if int(record['contracted_delivery_days']) < 1:
        return False, "contracted_delivery_days must be >= 1"
    valid_wcs = ['WC_MACH','WC_SHEET','WC_SMT','WC_ELEC','WC_SEEKER',
                 'WC_PROP','WC_WARHEAD','WC_INTEGRATION','WC_LAUNCHER',
                 'WC_PROOF','WC_TORPEDO','WC_QA_INSP','WC_PACK']
    if record['work_center_code'] not in valid_wcs:
        return False, f"Invalid work_center_code: {record['work_center_code']}"
    return True, "OK"

def auto_fill_record(record: dict) -> dict:
    rec = record.copy()
    
    # 1. Weapon System & Work Center Defaults
    weapon = rec.get('weapon_system') or 'Nag'
    rec['weapon_system'] = weapon
    
    wc = rec.get('work_center_code') or 'WC_SEEKER'
    rec['work_center_code'] = wc
    
    # 2. BDL Taxonomy Mappings
    unit_mappings = {
        'Nag': { 'manufacturing_unit': 'Hyderabad', 'cost_center': 'CC_HYD', 'pgl_no': 'PGL_NAG_ATGM', 'drdo_signoff_required': 1, 'indigenisation_pct': 85.0 },
        'Akash': { 'manufacturing_unit': 'Ibrahimpatnam', 'cost_center': 'CC_IBP', 'pgl_no': 'PGL_AKASH_SAM', 'drdo_signoff_required': 1, 'indigenisation_pct': 90.0 },
        'Milan2T': { 'manufacturing_unit': 'Bhanur', 'cost_center': 'CC_BHANUR', 'pgl_no': 'PGL_MILAN_ATGM', 'drdo_signoff_required': 0, 'indigenisation_pct': 60.0 },
        'Konkurs': { 'manufacturing_unit': 'Bhanur', 'cost_center': 'CC_BHANUR', 'pgl_no': 'PGL_KONKURS_ATGM', 'drdo_signoff_required': 0, 'indigenisation_pct': 55.0 },
        'Astra': { 'manufacturing_unit': 'Hyderabad', 'cost_center': 'CC_HYD', 'pgl_no': 'PGL_ASTRA_AAM', 'drdo_signoff_required': 1, 'indigenisation_pct': 80.0 },
        'HWT': { 'manufacturing_unit': 'Visakhapatnam', 'cost_center': 'CC_VIZAG', 'pgl_no': 'PGL_TORPEDO_HWT', 'drdo_signoff_required': 1, 'indigenisation_pct': 75.0 },
        'LWT': { 'manufacturing_unit': 'Visakhapatnam', 'cost_center': 'CC_VIZAG', 'pgl_no': 'PGL_TORPEDO_LWT', 'drdo_signoff_required': 1, 'indigenisation_pct': 70.0 },
        'Amogha3': { 'manufacturing_unit': 'Hyderabad', 'cost_center': 'CC_HYD', 'pgl_no': 'PGL_AMOGHA_ATGM', 'drdo_signoff_required': 1, 'indigenisation_pct': 92.0 },
        'HELINA': { 'manufacturing_unit': 'Hyderabad', 'cost_center': 'CC_HYD', 'pgl_no': 'PGL_HELINA_ATGM', 'drdo_signoff_required': 1, 'indigenisation_pct': 88.0 },
        'CMDS': { 'manufacturing_unit': 'Bhanur', 'cost_center': 'CC_BHANUR', 'pgl_no': 'PGL_CMDS', 'drdo_signoff_required': 0, 'indigenisation_pct': 95.0 }
    }
    
    wc_mappings = {
        'WC_MACH': { 'stage': 'Mech_Fab', 'seq': 1, 'op_time': 45.0, 'setup': 45.0 },
        'WC_SHEET': { 'stage': 'Mech_Fab', 'seq': 2, 'op_time': 35.0, 'setup': 30.0 },
        'WC_SMT': { 'stage': 'Electronics_Assy', 'seq': 3, 'op_time': 120.0, 'setup': 60.0 },
        'WC_ELEC': { 'stage': 'Electronics_Assy', 'seq': 4, 'op_time': 180.0, 'setup': 45.0 },
        'WC_SEEKER': { 'stage': 'Seeker_Assy', 'seq': 5, 'op_time': 300.0, 'setup': 90.0 },
        'WC_PROP': { 'stage': 'Propulsion', 'seq': 6, 'op_time': 200.0, 'setup': 120.0 },
        'WC_WARHEAD': { 'stage': 'Warhead', 'seq': 7, 'op_time': 150.0, 'setup': 90.0 },
        'WC_TORPEDO': { 'stage': 'Integration', 'seq': 8, 'op_time': 600.0, 'setup': 180.0 },
        'WC_INTEGRATION': { 'stage': 'Integration', 'seq': 9, 'op_time': 360.0, 'setup': 120.0 },
        'WC_LAUNCHER': { 'stage': 'Integration', 'seq': 10, 'op_time': 400.0, 'setup': 90.0 },
        'WC_PROOF': { 'stage': 'Proof_Test', 'seq': 11, 'op_time': 240.0, 'setup': 60.0 },
        'WC_QA_INSP': { 'stage': 'QA_Inspect', 'seq': 12, 'op_time': 90.0, 'setup': 15.0 },
        'WC_PACK': { 'stage': 'QA_Inspect', 'seq': 13, 'op_time': 45.0, 'setup': 15.0 }
    }
    
    # Apply weapon unit mappings
    w_map = unit_mappings.get(weapon, unit_mappings['Nag'])
    for key in ['manufacturing_unit', 'cost_center', 'pgl_no', 'drdo_signoff_required', 'indigenisation_pct']:
        if rec.get(key) is None or rec.get(key) == '':
            rec[key] = w_map[key]
            
    # Apply work center mappings
    wc_map = wc_mappings.get(wc, wc_mappings['WC_SEEKER'])
    if rec.get('sub_assembly_stage') is None or rec.get('sub_assembly_stage') == '':
        rec['sub_assembly_stage'] = wc_map['stage']
    if rec.get('operation_sequence') is None or rec.get('operation_sequence') == '':
        rec['operation_sequence'] = wc_map['seq']
    if rec.get('operation_time_min') is None or rec.get('operation_time_min') == '':
        rec['operation_time_min'] = wc_map['op_time']
    if rec.get('setup_time_min') is None or rec.get('setup_time_min') == '':
        rec['setup_time_min'] = wc_map['setup']
    if rec.get('machine_code') is None or rec.get('machine_code') == '':
        rec['machine_code'] = f"{wc}_LINE_1"
        
    # Process sheet details
    sheet_type = rec.get('process_sheet_type') or 'Standard'
    rec['process_sheet_type'] = sheet_type
    
    if rec.get('fai_required_flag') is None or rec.get('fai_required_flag') == '':
        rec['fai_required_flag'] = 1 if sheet_type == 'FAI' else 0
    fai_flag = int(rec['fai_required_flag'])
    
    # Shift and Parallel lines
    shift = rec.get('shift_pattern') or 'Single'
    rec['shift_pattern'] = shift
    shift_mult = 1 if shift == 'Single' else (2 if shift == 'Double' else 3)
    
    if rec.get('num_parallel_lines') is None or rec.get('num_parallel_lines') == '':
        rec['num_parallel_lines'] = 1
    num_parallel = int(rec['num_parallel_lines'])
    
    if rec.get('available_machine_hrs_day') is None or rec.get('available_machine_hrs_day') == '':
        rec['available_machine_hrs_day'] = float(8 * shift_mult)
        
    # Machine age and OEE
    unit = rec.get('manufacturing_unit')
    if rec.get('machine_age_years') is None or rec.get('machine_age_years') == '':
        if unit == 'Hyderabad': age = 22
        elif unit == 'Bhanur': age = 12
        elif unit == 'Visakhapatnam': age = 8
        else: age = 2
        rec['machine_age_years'] = age
        
    if rec.get('machine_oee_pct') is None or rec.get('machine_oee_pct') == '':
        if unit == 'Hyderabad': oee_base = 0.65
        elif unit == 'Bhanur': oee_base = 0.78
        elif unit == 'Visakhapatnam': oee_base = 0.82
        else: oee_base = 0.88
        rec['machine_oee_pct'] = oee_base
        
    if rec.get('planning_period_qty') is None or rec.get('planning_period_qty') == '':
        rec['planning_period_qty'] = 18
    plan_qty = int(rec['planning_period_qty'])
    
    if rec.get('working_days_in_period') is None or rec.get('working_days_in_period') == '':
        rec['working_days_in_period'] = 22
        
    if rec.get('contracted_delivery_days') is None or rec.get('contracted_delivery_days') == '':
        rec['contracted_delivery_days'] = 60
    delivery_days = int(rec['contracted_delivery_days'])
    
    if rec.get('vendor_lead_time_days') is None or rec.get('vendor_lead_time_days') == '':
        rec['vendor_lead_time_days'] = 120
        
    if rec.get('rework_rate_pct') is None or rec.get('rework_rate_pct') == '':
        rec['rework_rate_pct'] = 0.05
        
    if rec.get('export_order_flag') is None or rec.get('export_order_flag') == '':
        rec['export_order_flag'] = 1 if weapon == 'LWT' else 0
        
    drdo_required = int(rec['drdo_signoff_required'])
    
    # Lead times
    if rec.get('process_sheet_prep_days') is None or rec.get('process_sheet_prep_days') == '':
        if sheet_type == 'FAI': prep = 32.5
        elif sheet_type == 'Rework': prep = 5.5
        elif sheet_type in ['Life_Extension', 'Refurbishment']: prep = 22.5
        else: prep = 10.0
        rec['process_sheet_prep_days'] = prep
        
    if rec.get('drdo_signoff_lead_days') is None or rec.get('drdo_signoff_lead_days') == '':
        if drdo_required == 1:
            lead = 45.0 if sheet_type == 'FAI' else 20.0
        else:
            lead = 0.0
        rec['drdo_signoff_lead_days'] = lead
        
    if rec.get('jig_fixture_prep_days') is None or rec.get('jig_fixture_prep_days') == '':
        jig = 22.5 if wc in ['WC_MACH', 'WC_SMT', 'WC_SEEKER', 'WC_PROP'] else 10.0
        if fai_flag == 1:
            jig += 15.0
        rec['jig_fixture_prep_days'] = jig
        
    if rec.get('fai_setup_lead_days') is None or rec.get('fai_setup_lead_days') == '':
        rec['fai_setup_lead_days'] = 25.0 if fai_flag == 1 else 0.0
        
    # Derived properties
    op_time = float(rec.get('operation_time_min', 300.0))
    rec['total_smh'] = float((op_time * plan_qty) / 60.0)
    rec['delivery_urgency_score'] = 1.0 / (float(delivery_days) + 1.0)
    
    if rec.get('planning_period') is None or rec.get('planning_period') == '':
        rec['planning_period'] = '2023-01'
        
    # Cast variables to match type validation
    rec['machine_oee_pct'] = float(rec['machine_oee_pct'])
    rec['num_parallel_lines'] = int(rec['num_parallel_lines'])
    rec['planning_period_qty'] = int(rec['planning_period_qty'])
    rec['working_days_in_period'] = int(rec.get('working_days_in_period') or 22)
    rec['contracted_delivery_days'] = int(rec['contracted_delivery_days'])
    rec['fai_required_flag'] = int(rec['fai_required_flag'])
    rec['drdo_signoff_required'] = int(rec['drdo_signoff_required'])
    rec['vendor_lead_time_days'] = int(rec['vendor_lead_time_days'])
    rec['indigenisation_pct'] = float(rec['indigenisation_pct'])
    rec['rework_rate_pct'] = float(rec['rework_rate_pct'])
    rec['available_machine_hrs_day'] = float(rec['available_machine_hrs_day'])
    rec['operation_time_min'] = float(rec['operation_time_min'])
    rec['setup_time_min'] = float(rec['setup_time_min'])
    rec['machine_age_years'] = int(rec.get('machine_age_years') or 15)
    rec['export_order_flag'] = int(rec.get('export_order_flag') or 0)
    rec['operation_sequence'] = int(rec.get('operation_sequence') or 1)
    rec['process_sheet_prep_days'] = float(rec.get('process_sheet_prep_days') or 0.0)
    rec['drdo_signoff_lead_days'] = float(rec.get('drdo_signoff_lead_days') or 0.0)
    rec['jig_fixture_prep_days'] = float(rec.get('jig_fixture_prep_days') or 0.0)
    rec['fai_setup_lead_days'] = float(rec.get('fai_setup_lead_days') or 0.0)
    rec['planned_downtime_hrs_month'] = float(rec.get('planned_downtime_hrs_month') or 15.0)
    rec['skilled_tech_available'] = int(rec.get('skilled_tech_available') or 2)
    rec['qa_gate_clearance_hrs'] = float(rec.get('qa_gate_clearance_hrs') or 4.0)
    rec['contract_order_qty'] = int(rec.get('contract_order_qty') or (rec['planning_period_qty'] + 180))
    
    return rec

def translate_shap_contribution(col, val) -> str:
    # Decode label-encoded categorical values if encoders are available
    global preprocessors
    if preprocessors is not None and 'encoders' in preprocessors and col in preprocessors['encoders']:
        try:
            le = preprocessors['encoders'][col]
            val_int = int(float(val))
            if 0 <= val_int < len(le.classes_):
                val = le.classes_[val_int]
        except Exception:
            pass

    # 1. routing_criticality
    if col == 'routing_criticality':
        if float(val) >= 0.5:
            return "This is a critical path Work Center (Seeker/Integration/Proof Test)"
        return "Standard path Work Center (non-critical stage)"
    # 2. load_per_line
    elif col == 'load_per_line':
        load = float(val)
        if load > 150.0:
            return f"Load per assembly line = {load:.1f} hrs (very high)"
        elif load > 80.0:
            return f"Load per assembly line = {load:.1f} hrs (moderate)"
        return f"Load per assembly line = {load:.1f} hrs (low)"
    # 3. required_capacity_hrs
    elif col == 'required_capacity_hrs':
        hrs = float(val)
        if hrs > 300.0:
            return f"Total required capacity = {hrs:.1f} hrs (exceeds single-shift limits)"
        return f"Total required capacity = {hrs:.1f} hrs (manageable)"
    # 4. fai_required_flag
    elif col == 'fai_required_flag':
        if float(val) >= 0.5:
            return "First Article Inspection required — adds 30% time overhead"
        return "First Article Inspection (FAI) not required"
    # 5. drdo_signoff_required
    elif col == 'drdo_signoff_required':
        if float(val) >= 0.5:
            return "DRDO Sign-off required — adds coordination lead time buffer"
        return "No DRDO Sign-off required for this batch"
    # 6. rework_rate_pct
    elif col == 'rework_rate_pct':
        rate = float(val) * 100.0
        if rate > 8.0:
            return f"Rework rate of {rate:.1f}% contributes to cumulative capacity wastage (critical)"
        return f"Rework rate of {rate:.1f}% is within acceptable operating limits"
    # 7. machine_oee_pct
    elif col == 'machine_oee_pct':
        oee = float(val) * 100.0
        if oee < 70.0:
            return f"Machine operating efficiency (OEE) is low at {oee:.1f}%"
        return f"Machine OEE is optimal at {oee:.1f}%"
    # 8. machine_age_years
    elif col == 'machine_age_years':
        age = int(val)
        if age > 20:
            return f"Aging machine ({age} years old) increases breakdown risk"
        return f"Machine age ({age} years) is within standard lifespan"
    # 9. planning_period_qty
    elif col == 'planning_period_qty':
        qty = int(val)
        if qty > 25:
            return f"High planning period quantity of {qty} units drives up required capacity"
        return f"Planning period quantity is standard ({qty} units)"
    # 10. contracted_delivery_days
    elif col == 'contracted_delivery_days':
        days = int(val)
        if days < 90:
            return f"Tight delivery window ({days} days) increases risk of scheduling bottlenecks"
        return f"Comfortable delivery window ({days} days)"
    # Fallback to general translation
    else:
        nice_name = SHAP_TRANSLATION_MAP.get(col, col)
        if isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)
        return f"{nice_name}: {val_str}"

def generate_recommendations(record: dict, prediction: dict) -> list[str]:
    recs = []
    util = prediction.get('utilization_rate', 0.0)
    
    # 1. Capacity / Shift constraints
    if util > 0.85:
        shift = record.get('shift_pattern', 'Single')
        parallel = int(record.get('num_parallel_lines') or 1)
        qty = int(record.get('planning_period_qty') or 1)
        
        if shift == 'Single':
            recs.append("Increase shift pattern to Double or Triple to expand available operating hours.")
        elif shift == 'Double':
            recs.append("Increase shift pattern to Triple to maximize daily operating time.")
            
        if parallel < 4:
            recs.append(f"Deploy additional parallel lines (current: {parallel}) to distribute work center load.")
            
        if qty > 15:
            recs.append(f"Split planning period batch (current qty: {qty}) into smaller sub-lots to spread load across multiple periods.")
            
    # 2. First Article Inspection (FAI) constraints
    if int(record.get('fai_required_flag') or 0) == 1:
        delivery_days = int(record.get('contracted_delivery_days') or 60)
        if delivery_days < 60:
            recs.append("Initiate FAI documentation and tooling setup immediately; FAI setup requires up to 25 extra days.")
        else:
            recs.append("Ensure FAI process sheet approvals are tracked. Plan for 15-35 days FAI setup lead time.")
            
    # 3. DRDO Sign-off constraints
    if int(record.get('drdo_signoff_required') or 0) == 1:
        delivery_days = int(record.get('contracted_delivery_days') or 60)
        lead_days = float(record.get('drdo_signoff_lead_days') or 0.0)
        if delivery_days < 90:
            recs.append(f"Coordinate with DRDO representatives immediately to bypass potential sign-off delays (lead time: {lead_days:.1f} days).")
            
    # 4. Supply Chain / Vendor Lead Time constraints
    vendor_lead = int(record.get('vendor_lead_time_days') or 120)
    indig_pct = float(record.get('indigenisation_pct') or 80.0)
    if vendor_lead > 180 and indig_pct < 70.0:
        recs.append(f"High supply chain risk (vendor lead: {vendor_lead} days, indigenisation: {indig_pct}%). Fast-track local vendor development or source alternative indigenised materials.")
        
    # 5. Quality / Rework rate constraints
    rework = float(record.get('rework_rate_pct') or 0.0)
    if rework > 0.08:
        wc = record.get('work_center_code', 'Work Center')
        recs.append(f"High rework rate ({rework*100:.1f}%) detected at {wc}. Schedule preventative QA audits, inspect tooling calibration, and run training sessions.")
        
    # 6. Machine reliability constraints
    age = int(record.get('machine_age_years') or 0)
    oee = float(record.get('machine_oee_pct') or 0.8)
    if oee < 0.70 or age > 20:
        recs.append(f"Plan preventative machine maintenance or overhaul; machine OEE is low at {oee*100:.1f}% and age is {age} years.")
        
    # 7. Fallback / General recommendations
    if len(recs) == 0:
        recs.append("Production capacity is sufficient. Maintain standard setup times and schedule standard line monitoring.")
        weapon = record.get('weapon_system', 'Weapon')
        recs.append(f"Optimize inventory buffers for {weapon} components to match the planned timeline.")
        
    return recs[:5]

def predict_capacity(record: dict) -> dict:
    global features_other, explainer_C
    init_models()
    
    # Auto-fill missing/optional values to build a complete record first
    filled_record = auto_fill_record(record)
    
    # Enhancement 2: Custom validation function call
    is_valid, err_msg = validate_record(filled_record)
    if not is_valid:
        raise ValueError(err_msg)
        
    # Convert validated record dict to DataFrame
    df = pd.DataFrame([filled_record])
    
    # Custom Feature Engineering
    df['effective_demand'] = df['planning_period_qty'] / (1.0 - df['rework_rate_pct'])
    df['urgency_score'] = 1.0 / (df['contracted_delivery_days'] + 1.0)
    df['oee_age_factor'] = df['machine_oee_pct'] * (1.0 - 0.005 * df['machine_age_years'])
    df['supply_risk_score'] = df['vendor_lead_time_days'] * (1.0 - df['indigenisation_pct'] / 100.0)
    df['routing_criticality'] = df['work_center_code'].apply(
        lambda x: 1 if x in ['WC_SEEKER', 'WC_INTEGRATION', 'WC_PROOF'] else 0
    )
    
    # Categorical and numerical column lists
    categorical_cols = preprocessors['categorical_cols']
    numerical_cols = [c for c in preprocessors['numerical_cols'] if c not in ['load_per_line', 'required_capacity_hrs']]
    encoders = preprocessors['encoders']
    clip_thresholds = preprocessors['clip_thresholds']
    
    # Apply Label Encoding
    df_encoded = df.copy()
    for col in categorical_cols:
        le = encoders[col]
        val = str(df_encoded.iloc[0][col])
        if val in le.classes_:
            df_encoded[col] = le.transform([val])[0]
        else:
            df_encoded[col] = le.transform([le.classes_[0]])[0]
            
    # Apply clipping thresholds
    for col in numerical_cols:
        q01, q99 = clip_thresholds[col]
        df_encoded[col] = df_encoded[col].clip(q01, q99)
        
    # Feature list A
    features_A = categorical_cols + numerical_cols
    X_A = df_encoded[features_A]
    
    # Predict required_capacity_hrs (Model A)
    pred_capacity = float(models['A'].predict(X_A)[0])
    pred_capacity = max(1.0, pred_capacity)
    
    # Add predicted capacity and load per line as features for subsequent models
    df_encoded['required_capacity_hrs'] = pred_capacity
    df_encoded['load_per_line'] = pred_capacity / float(df_encoded.iloc[0]['num_parallel_lines'])
    
    # Clip engineered features dependent on required capacity
    for col in ['required_capacity_hrs', 'load_per_line']:
        if col in clip_thresholds:
            q01, q99 = clip_thresholds[col]
            df_encoded[col] = df_encoded[col].clip(q01, q99)
        
    X_other = df_encoded[features_other]
    
    # Predict utilization rate (Model B)
    pred_util = float(models['B'].predict(X_A)[0])
    pred_util = max(0.0, pred_util)
    
    # Predict bottleneck flag (Model C) and probability
    pred_bottleneck = int(models['C'].predict(X_A)[0])
    bottleneck_prob = float(models['C'].predict_proba(X_A)[0, 1])
    
    # Predict delivery risk flag (Model D) and probability
    delivery_risk_prob = float(models['D'].predict_proba(X_other)[0, 1])
    pred_delivery_risk = int(delivery_risk_prob >= 0.35)
    
    # Predict overload severity (Model E) and probabilities
    pred_severity_idx = int(models['E'].predict(X_A)[0])
    severity_labels = {0: 'OK', 1: 'Warning', 2: 'Critical'}
    pred_severity = severity_labels.get(pred_severity_idx, 'OK')
    
    severity_probs_raw = models['E'].predict_proba(X_A)[0]
    
    # Compute SHAP top 3 reasons for bottleneck flag (with fail-safe fallback)
    shap_top3 = []
    try:
        if explainer_C is not None and features_A is not None:
            shap_vals = explainer_C(X_A)
            contribs = []
            for j, col in enumerate(features_A):
                col_val = df.iloc[0][col] if col in df.columns else df_encoded.iloc[0][col]
                shap_val = shap_vals.values[0, j]
                contribs.append((col, col_val, shap_val))
                
            # Sort by SHAP value descending (most positive contributions first)
            contribs = sorted(contribs, key=lambda x: x[2], reverse=True)
            
            for col, val, s_val in contribs[:3]:
                shap_top3.append(translate_shap_contribution(col, val))
        else:
            raise ValueError("explainer_C or features_A is not initialized")
    except Exception as shap_err:
        print(f"SHAP explanation failed: {shap_err}. Using fallback reasons.")
        shap_top3 = [
            f"Line Load (hrs): {pred_capacity / float(filled_record.get('num_parallel_lines', 1)):.2f}",
            f"Required Capacity: {pred_capacity:.2f} hrs",
            f"Work Center: {filled_record.get('work_center_code', 'Unknown')}"
        ]
            
    # Generate BDL-specific Recommendations
    recommendations = generate_recommendations(filled_record, {
        'required_capacity_hrs': pred_capacity,
        'utilization_rate': pred_util,
        'bottleneck_flag': pred_bottleneck
    })
    
    return {
        'required_capacity_hrs': round(pred_capacity, 2),
        'utilization_rate': round(pred_util, 4),
        'bottleneck_flag': pred_bottleneck,
        'bottleneck_probability': round(bottleneck_prob, 4),
        'delivery_risk_flag': pred_delivery_risk,
        'delivery_risk_probability': round(delivery_risk_prob, 4),
        'overload_severity': pred_severity,
        'severity_probs': {
            'OK': round(float(severity_probs_raw[0]), 4),
            'Warning': round(float(severity_probs_raw[1]), 4),
            'Critical': round(float(severity_probs_raw[2]), 4)
        },
        'shap_top3_reasons': shap_top3,
        'recommendations': recommendations,
        'validated_record': filled_record  # Keep track of filled metadata fields
    }
