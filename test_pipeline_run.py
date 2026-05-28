import pandas as pd
import json
from inference import predict_capacity

def run_test():
    print("Loading sample records...")
    df = pd.read_csv('sample_records.csv')
    
    # Take first row as dict
    sample = df.iloc[0].to_dict()
    
    # Print the input sample (omitting target variables)
    print("Sample Input Record (first 5 features):")
    sample_input = {k: v for k, v in sample.items() if k not in ['required_capacity_hrs', 'utilization_rate', 'bottleneck_flag', 'delivery_risk_flag', 'overload_severity']}
    print(json.dumps(list(sample_input.items())[:5], indent=2))
    
    print("\nRunning predict_capacity inference pipeline...")
    prediction = predict_capacity(sample_input)
    
    print("\nPrediction Output:")
    print(json.dumps(prediction, indent=2))
    
    # Assertions
    assert 'required_capacity_hrs' in prediction
    assert 'utilization_rate' in prediction
    assert 'bottleneck_flag' in prediction
    assert 'bottleneck_probability' in prediction
    assert 'delivery_risk_flag' in prediction
    assert 'delivery_risk_probability' in prediction
    assert 'overload_severity' in prediction
    assert 'severity_probs' in prediction
    assert 'shap_top3_reasons' in prediction
    
    # Assert probabilities are valid floats between 0 and 1
    assert 0.0 <= prediction['bottleneck_probability'] <= 1.0
    assert 0.0 <= prediction['delivery_risk_probability'] <= 1.0
    assert 0.0 <= prediction['severity_probs']['OK'] <= 1.0
    assert 0.0 <= prediction['severity_probs']['Warning'] <= 1.0
    assert 0.0 <= prediction['severity_probs']['Critical'] <= 1.0
    
    print("\nInference pipeline verification SUCCESS!")

if __name__ == '__main__':
    run_test()
