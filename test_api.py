import urllib.request
import json
import pandas as pd

def test_api():
    print("Testing REST API /predict endpoint...")
    
    # Load sample records
    df = pd.read_csv('sample_records.csv')
    sample = df.iloc[0].to_dict()
    
    # Remove target columns
    sample_input = {k: v for k, v in sample.items() if k not in ['required_capacity_hrs', 'utilization_rate', 'bottleneck_flag', 'delivery_risk_flag', 'overload_severity']}
    
    # Convert to JSON bytes
    data = json.dumps(sample_input).encode('utf-8')
    
    # Send request
    req = urllib.request.Request(
        'http://127.0.0.1:5050/predict', 
        data=data, 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            print("API Response:")
            print(json.dumps(res_json, indent=2))
            print("\nAPI Integration Test: SUCCESS!")
    except Exception as e:
        print(f"API Integration Test: FAILED! Error: {str(e)}")

if __name__ == '__main__':
    test_api()
