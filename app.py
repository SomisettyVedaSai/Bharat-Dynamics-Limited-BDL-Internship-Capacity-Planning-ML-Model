import os
import io
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file, render_template_string, make_response
import pickle
import plotly.express as px
import plotly.io as pio
from inference import predict_capacity, init_models
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from pymongo import MongoClient
from bson.objectid import ObjectId
import json
import time
import logging

# Enhancement 5: Configure python logging throughout (dual log files)
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('bdl_app.log'),
        logging.FileHandler(os.path.join('logs', 'bdl_system_audit.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('bdl_app')

app = Flask(__name__)
# Bug 5: Load BDL_SECRET_KEY from environment with fallback
# Set BDL_SECRET_KEY environment variable in production
app.secret_key = os.environ.get('BDL_SECRET_KEY', 'bdl_dev_fallback_key_change_in_prod')

# Initialize MongoDB client with fallback
try:
    mongo_client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
    # Check connection
    mongo_client.server_info()
    db = mongo_client['bdl_capacity_planning']
    users_col = db['users']
    predictions_col = db['predictions']
    mongo_active = True
    logger.info("MongoDB connection established")
except Exception as e:
    logger.warning("MongoDB not running or accessible. Falling back to local storage files. Error: %s", str(e))
    mongo_active = False

# Fallback Local Storage functions
def get_users_fallback():
    users_file = os.path.join('data', 'local_users.json')
    if os.path.exists(users_file):
        try:
            with open(users_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_users_fallback(users):
    users_file = os.path.join('data', 'local_users.json')
    os.makedirs(os.path.dirname(users_file), exist_ok=True)
    with open(users_file, 'w') as f:
        json.dump(users, f, indent=4)

def get_predictions_fallback(username):
    preds_file = os.path.join('data', 'local_predictions.json')
    if os.path.exists(preds_file):
        try:
            with open(preds_file, 'r') as f:
                all_preds = json.load(f)
                return [p for p in all_preds if p.get('username') == username]
        except Exception:
            return []
    return []

def save_prediction_fallback(username, record, prediction):
    preds_file = os.path.join('data', 'local_predictions.json')
    os.makedirs(os.path.dirname(preds_file), exist_ok=True)
    all_preds = []
    if os.path.exists(preds_file):
        try:
            with open(preds_file, 'r') as f:
                all_preds = json.load(f)
        except Exception:
            pass
            
    pred_entry = {
        '_id': str(time.time()),
        'username': username,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'record': record,
        'prediction': prediction
    }
    all_preds.append(pred_entry)
    with open(preds_file, 'w') as f:
        json.dump(all_preds, f, indent=4)
    return pred_entry

def delete_prediction_fallback(username, pred_id):
    preds_file = os.path.join('data', 'local_predictions.json')
    if os.path.exists(preds_file):
        try:
            with open(preds_file, 'r') as f:
                all_preds = json.load(f)
            new_preds = [p for p in all_preds if not (p.get('_id') == pred_id and p.get('username') == username)]
            with open(preds_file, 'w') as f:
                json.dump(new_preds, f, indent=4)
            return True
        except Exception:
            return False
    return False

# Database wrappers
def db_find_user(username):
    if mongo_active:
        return users_col.find_one({'username': username})
    else:
        users = get_users_fallback()
        if username in users:
            return {'username': username, 'password': users[username]['password']}
        return None

def db_create_user(username, hashed_password):
    if mongo_active:
        users_col.insert_one({'username': username, 'password': hashed_password})
    else:
        users = get_users_fallback()
        users[username] = {'password': hashed_password}
        save_users_fallback(users)

def db_save_prediction(username, record, prediction):
    pred_copy = prediction.copy()
    if 'validated_record' in pred_copy:
        del pred_copy['validated_record']
        
    if mongo_active:
        entry = {
            'username': username,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'record': record,
            'prediction': pred_copy
        }
        predictions_col.insert_one(entry)
        entry['_id'] = str(entry['_id'])
        return entry
    else:
        return save_prediction_fallback(username, record, pred_copy)

def db_get_predictions(username):
    if mongo_active:
        preds = list(predictions_col.find({'username': username}).sort('timestamp', -1))
        for p in preds:
            p['_id'] = str(p['_id'])
        return preds
    else:
        preds = get_predictions_fallback(username)
        return sorted(preds, key=lambda x: x['timestamp'], reverse=True)

def db_delete_prediction(username, pred_id):
    if mongo_active:
        try:
            res = predictions_col.delete_one({'_id': ObjectId(pred_id), 'username': username})
            return res.deleted_count > 0
        except Exception:
            return False
    else:
        return delete_prediction_fallback(username, pred_id)

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Set Plotly template
pio.templates.default = "plotly_white"

# Path to the data
DATA_PATH = os.path.join('data', 'bdl_production_planning_data.csv')

def get_dashboard_data():
    if not os.path.exists(DATA_PATH):
        return None
        
    df = pd.read_csv(DATA_PATH)
    
    # Compute summary stats
    total_records = len(df)
    bottleneck_count = int(df['bottleneck_flag'].sum())
    delivery_risk_count = int(df['delivery_risk_flag'].sum())
    critical_count = int((df['overload_severity'] == 'Critical').sum())
    
    # Active alerts: Critical overload AND delivery days < 90
    active_alerts = df[(df['overload_severity'] == 'Critical') & (df['contracted_delivery_days'] < 90)]
    alert_count = len(active_alerts)
    
    # Generate Heatmap of Work Center x Weapon System Utilization
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
        labels=dict(x="Weapon System", y="Work Center", color="Avg Utilization"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Work Center × Weapon System Average Capacity Utilization"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Outfit, sans-serif", size=12),
        coloraxis_colorbar=dict(title="Utilization", tickformat=".0%")
    )
    heatmap_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    # Generate Month-on-Month Required Capacity Trend by Weapon System
    trend_df = df.groupby(['planning_period', 'weapon_system'])['required_capacity_hrs'].mean().reset_index()
    fig_trend = px.line(
        trend_df,
        x='planning_period',
        y='required_capacity_hrs',
        color='weapon_system',
        title="Month-on-Month Average Required Capacity Load by Weapon System"
    )
    fig_trend.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Outfit, sans-serif", size=12),
        xaxis_title="Planning Period",
        yaxis_title="Required Capacity (Hours)"
    )
    trend_html = pio.to_html(fig_trend, full_html=False, include_plotlyjs='cdn')
    
    # Prepare Alerts List (top 10 for dashboard display)
    alerts_list = active_alerts.sort_values('utilization_rate', ascending=False).head(10).to_dict('records')
    
    # Top 15 bottlenecked work-center-weapon combos
    top_utilized = df.sort_values('utilization_rate', ascending=False).head(15).to_dict('records')
    
    return {
        'total_records': total_records,
        'bottleneck_count': bottleneck_count,
        'delivery_risk_count': delivery_risk_count,
        'critical_count': critical_count,
        'alert_count': alert_count,
        'heatmap_html': heatmap_html,
        'trend_html': trend_html,
        'alerts_list': alerts_list,
        'top_utilized': top_utilized
    }

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('dashboard'))
        
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        user = db_find_user(username)
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            logger.info("User '%s' successfully logged in at timestamp %s", username, time.strftime('%Y-%m-%d %H:%M:%S'))
            return redirect(url_for('dashboard'))
        else:
            logger.warning("Failed login attempt for username '%s' at timestamp %s", username, time.strftime('%Y-%m-%d %H:%M:%S'))
            error = "Invalid username or password"
            
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('dashboard'))
        
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not password:
            error = "All fields are required"
        elif password != confirm_password:
            error = "Passwords do not match"
        elif db_find_user(username):
            error = "Username already exists"
        else:
            hashed_pw = generate_password_hash(password)
            db_create_user(username, hashed_pw)
            logger.info("User '%s' registered successfully at timestamp %s", username, time.strftime('%Y-%m-%d %H:%M:%S'))
            return render_template('login.html', success="Registration successful! Please log in.")
            
    return render_template('register.html', error=error)

@app.route('/logout')
def logout():
    username = session.pop('username', None)
    if username:
        logger.info("User '%s' logged out at timestamp %s", username, time.strftime('%Y-%m-%d %H:%M:%S'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    username = session['username']
    data = get_dashboard_data()
    if data is None:
        return "Error: Please run dataset generation first (python data_generation.py) and training (python train_pipeline.py)."
        
    data['username'] = username
    data['mongo_active'] = mongo_active
    data['predictions_history'] = db_get_predictions(username)
    
    # Load evaluation report json to show model performance cards
    evaluation_report = {}
    report_path = os.path.join('models', 'evaluation_report.json')
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                evaluation_report = json.load(f)
        except Exception:
            pass
    data['evaluation_report'] = evaluation_report
    
    return render_template('dashboard.html', **data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        record = request.json
        if not record:
            return jsonify({'error': 'No input data provided'}), 400
            
        prediction = predict_capacity(record)
        return jsonify(prediction)
    except Exception as e:
        logger.error("API predict endpoint error: %s", str(e))
        return jsonify({'error': str(e)}), 500
@app.route('/predict_ui', methods=['POST'])
@login_required
def predict_ui():
    try:
        username = session['username']
        
        if request.is_json:
            input_data = request.json
        else:
            input_data = request.form.to_dict()
            
        record = {}
        for k, v in input_data.items():
            if v == "" or v is None:
                record[k] = None
            else:
                try:
                    v_str = str(v)
                    if '.' in v_str:
                        record[k] = float(v_str)
                    else:
                        record[k] = int(v_str)
                except ValueError:
                    record[k] = v
        
        delivery_days = int(record.get('contracted_delivery_days') or 60)
        if delivery_days < 1:
            delivery_days = 1
        record['contracted_delivery_days'] = delivery_days
        record['delivery_urgency_score'] = 1.0 / (float(delivery_days) + 1.0)
        
        prediction = predict_capacity(record)
        validated_rec = prediction.get('validated_record', record)
        
        # Save prediction run
        db_save_prediction(username, validated_rec, prediction)
        logger.info("Prediction run via UI by '%s' for weapon '%s' at '%s' at timestamp %s", 
                    username, validated_rec.get('weapon_system'), validated_rec.get('work_center_code'), 
                    time.strftime('%Y-%m-%d %H:%M:%S'))
        
        data = get_dashboard_data()
        if data is None:
            return jsonify({'success': False, 'error': 'Failed to load dashboard data'}), 500
            
        predictions_history = db_get_predictions(username)
        
        # Always return JSON for AJAX compatibility and caching safety
        alerts_html = render_template('alerts_template.html', alerts_list=data['alerts_list'])
        history_html = render_template('history_template.html', predictions_history=predictions_history)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'kpis': {
                'total_records': data['total_records'],
                'bottleneck_count': data['bottleneck_count'],
                'alert_count': data['alert_count'],
                'critical_count': data['critical_count']
            },
            'alerts_html': alerts_html,
            'history_html': history_html,
            'heatmap_html': data['heatmap_html']
        })
            
    except Exception as e:
        logger.error("UI predict endpoint error: %s", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict_ui_json', methods=['POST'])
@login_required
def predict_ui_json():
    return predict_ui()

# Enhancement 3: Add the JSON batch prediction endpoint
@app.route('/batch_predict', methods=['POST'])
@login_required
def batch_predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        df = pd.read_csv(file)
        # Drop target columns if present
        target_cols = ['required_capacity_hrs', 'utilization_rate',
                       'bottleneck_flag', 'delivery_risk_flag', 'overload_severity']
        df = df.drop(columns=[c for c in target_cols if c in df.columns])
        results = []
        for _, row in df.iterrows():
            record = row.to_dict()
            pred = predict_capacity(record)
            pred['input'] = record
            results.append(pred)
        logger.info("Batch prediction endpoint called by '%s' at timestamp %s for %d rows",
                    session['username'], time.strftime('%Y-%m-%d %H:%M:%S'), len(df))
        return jsonify({'count': len(results), 'predictions': results})
    except Exception as e:
        logger.error("Batch prediction endpoint error: %s", str(e))
        return jsonify({'error': str(e)}), 500

# Keep CSV predict batch route for UI compatibility
@app.route('/predict_batch', methods=['POST'])
@login_required
def predict_batch():
    try:
        username = session['username']
        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400
            
        df = pd.read_csv(file)
        target_cols = ['required_capacity_hrs', 'utilization_rate',
                       'bottleneck_flag', 'delivery_risk_flag', 'overload_severity']
        df = df.drop(columns=[c for c in target_cols if c in df.columns])
        
        predictions_list = []
        for idx, row in df.iterrows():
            record_dict = row.to_dict()
            try:
                pred = predict_capacity(record_dict)
                out_row = pred['validated_record'].copy()
                out_row['predicted_required_capacity_hrs'] = pred['required_capacity_hrs']
                out_row['predicted_utilization_rate'] = pred['utilization_rate']
                out_row['predicted_bottleneck_flag'] = pred['bottleneck_flag']
                out_row['predicted_bottleneck_probability'] = pred['bottleneck_probability']
                out_row['predicted_delivery_risk_flag'] = pred['delivery_risk_flag']
                out_row['predicted_delivery_risk_probability'] = pred['delivery_risk_probability']
                out_row['predicted_overload_severity'] = pred['overload_severity']
                out_row['predicted_severity_ok_probability'] = pred['severity_probs']['OK']
                out_row['predicted_severity_warning_probability'] = pred['severity_probs']['Warning']
                out_row['predicted_severity_critical_probability'] = pred['severity_probs']['Critical']
                out_row['shap_top3_reasons'] = " | ".join(pred['shap_top3_reasons'])
                predictions_list.append(out_row)
            except Exception as row_err:
                err_row = record_dict.copy()
                err_row['error_log'] = str(row_err)
                predictions_list.append(err_row)
                
        out_df = pd.DataFrame(predictions_list)
        buffer = io.BytesIO()
        out_df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        logger.info("Batch CSV prediction run successfully completed by user '%s' at timestamp %s", username, time.strftime('%Y-%m-%d %H:%M:%S'))
        return send_file(
            buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name='bdl_capacity_predictions.csv'
        )
    except Exception as e:
        logger.error("predict_batch route error: %s", str(e))
        return f"Error running batch prediction: {str(e)}", 500

# Enhancement 4: Export predictions history as CSV
@app.route('/export_predictions')
@login_required
def export_predictions():
    import io
    from flask import make_response
    username = session['username']
    preds = db_get_predictions(username)
    rows = []
    for p in preds:
        row = {'timestamp': p.get('timestamp', '')}
        # Flatten record fields
        rec = p.get('record', {})
        for k, v in rec.items():
            row[f"input_{k}"] = v
        # Flatten prediction fields
        pred = p.get('prediction', {}).copy()
        shap = pred.get('shap_top3_reasons', [])
        pred['shap_top3_reasons'] = ' | '.join(shap) if isinstance(shap, list) else str(shap)
        # Flatten severity probabilities
        sev_probs = pred.get('severity_probs', {})
        for k, v in sev_probs.items():
            pred[f"severity_prob_{k}"] = v
        if 'severity_probs' in pred:
            del pred['severity_probs']
            
        row.update(pred)
        rows.append(row)
        
    df_export = pd.DataFrame(rows)
    output = io.StringIO()
    df_export.to_csv(output, index=False)
    
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=bdl_predictions.csv'
    response.headers['Content-Type'] = 'text/csv'
    
    logger.info("Export predictions endpoint called by '%s' at timestamp %s", username, time.strftime('%Y-%m-%d %H:%M:%S'))
    return response

# Keep /export_history for UI link compatibility
@app.route('/export_history')
@login_required
def export_history():
    return export_predictions()

def get_routing_wcs(weapon):
    if weapon in ['HWT', 'LWT']:
        wcs = ['WC_MACH', 'WC_SHEET', 'WC_SMT', 'WC_ELEC', 'WC_TORPEDO', 'WC_PROOF', 'WC_QA_INSP', 'WC_PACK']
    else:
        wcs = ['WC_MACH', 'WC_SHEET', 'WC_ELEC', 'WC_PROP', 'WC_WARHEAD', 'WC_INTEGRATION', 'WC_PROOF', 'WC_QA_INSP', 'WC_PACK']
        if weapon in ['Akash', 'Nag', 'Astra', 'Amogha3', 'HELINA']:
            wcs.append('WC_SEEKER')
        if weapon in ['Akash', 'Nag', 'Astra', 'Amogha3', 'CMDS']:
            wcs.append('WC_SMT')
        if weapon in ['Akash', 'Milan2T', 'Konkurs', 'Nag', 'HELINA']:
            wcs.append('WC_LAUNCHER')
    
    seq_map = {
        'WC_MACH': 1, 'WC_SHEET': 2, 'WC_SMT': 3, 'WC_ELEC': 4, 'WC_SEEKER': 5,
        'WC_PROP': 6, 'WC_WARHEAD': 7, 'WC_TORPEDO': 8, 'WC_INTEGRATION': 9,
        'WC_LAUNCHER': 10, 'WC_PROOF': 11, 'WC_QA_INSP': 12, 'WC_PACK': 13
    }
    return sorted(wcs, key=lambda x: seq_map.get(x, 99))

@app.route('/predict_routing', methods=['POST'])
@login_required
def predict_routing():
    try:
        if request.is_json:
            input_data = request.json
        else:
            input_data = request.form.to_dict()
            
        weapon = input_data.get('weapon_system', 'Nag')
        qty = int(input_data.get('planning_period_qty') or 18)
        shift = input_data.get('shift_pattern', 'Single')
        parallel = int(input_data.get('num_parallel_lines') or 1)
        working_days = int(input_data.get('working_days_in_period') or 22)
        planning_period = input_data.get('planning_period', '2023-01')
        
        wcs = get_routing_wcs(weapon)
        
        default_parallel_lines_map = {
            'WC_MACH': 3,
            'WC_SHEET': 2,
            'WC_SMT': 2,
            'WC_ELEC': 2,
            'WC_SEEKER': 2,
            'WC_PROP': 1,
            'WC_WARHEAD': 1,
            'WC_TORPEDO': 1,
            'WC_INTEGRATION': 1,
            'WC_LAUNCHER': 1,
            'WC_PROOF': 1,
            'WC_QA_INSP': 2,
            'WC_PACK': 2
        }
        
        results = []
        for wc in wcs:
            wc_parallel = default_parallel_lines_map.get(wc, 1) * parallel
            rec = {
                'weapon_system': weapon,
                'work_center_code': wc,
                'planning_period_qty': qty,
                'shift_pattern': shift,
                'num_parallel_lines': wc_parallel,
                'working_days_in_period': working_days,
                'planning_period': planning_period
            }
            pred = predict_capacity(rec)
            results.append({
                'work_center_code': wc,
                'sequence': pred['validated_record'].get('operation_sequence', 99),
                'required_capacity_hrs': pred['required_capacity_hrs'],
                'utilization_rate': pred['utilization_rate'],
                'bottleneck_flag': pred['bottleneck_flag'],
                'overload_severity': pred['overload_severity']
            })
        return jsonify({'success': True, 'routing_predictions': results})
    except Exception as e:
        logger.error("Predict routing error: %s", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

def db_get_prediction_by_id(pred_id):
    if mongo_active:
        try:
            p = predictions_col.find_one({'_id': ObjectId(pred_id)})
            if p:
                p['_id'] = str(p['_id'])
                return p
        except Exception:
            return None
    else:
        preds_file = os.path.join('data', 'local_predictions.json')
        if os.path.exists(preds_file):
            try:
                with open(preds_file, 'r') as f:
                    all_preds = json.load(f)
                for p in all_preds:
                    if p.get('_id') == pred_id:
                        return p
            except Exception:
                return None
    return None

@app.route('/print_report/<pred_id>')
@login_required
def print_report(pred_id):
    pred_data = db_get_prediction_by_id(pred_id)
    if not pred_data:
        return "Error: Prediction record not found", 404
    return render_template('print_report.html', item=pred_data)

# =========================================================================
# ERP/SAP HANA PRODUCTION RUN CONNECTION HOOKS (BDL ENHANCEMENTS)
# In production, ML inputs are loaded directly from BDL's active database tables.
# Below is a commented connection hook blueprint showing pyrfc / RFC connection.
#
# from pyrfc import Connection
# def get_sap_contract_details(contract_id):
#     try:
#         conn = Connection(ashost='10.120.45.10', sysnr='01', client='400', user='ML_INTEGRATION', passwd='...')
#         res = conn.call('RFC_READ_TABLE', QUERY_TABLE='VBAK', OPTIONS=[{'TEXT': f"VBELN = '{contract_id}'"}])
#         return res
#     except Exception as e:
#         logger.error("Failed to query BDL SAP instance: %s", str(e))
#         return None
# =========================================================================

@app.route('/delete_prediction/<pred_id>', methods=['POST'])
@login_required
def delete_prediction(pred_id):
    username = session['username']
    success = db_delete_prediction(username, pred_id)
    if success:
        logger.info("User '%s' deleted prediction record ID '%s' at timestamp %s", 
                    username, pred_id, time.strftime('%Y-%m-%d %H:%M:%S'))
    else:
        logger.warning("User '%s' failed to delete prediction record ID '%s' at timestamp %s", 
                       username, pred_id, time.strftime('%Y-%m-%d %H:%M:%S'))
    return jsonify({'success': success})

if __name__ == '__main__':
    logger.info("Initializing BDL capacity models...")
    init_models()
    port = int(os.environ.get('BDL_PORT', 5050))
    logger.info(f"Starting Flask dashboard server on BDL internal server (http://127.0.0.1:{port})...")
    app.run(debug=True, port=port)
