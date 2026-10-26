import os
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("⚡ Building Machine Learning Pipeline...")

# --- 1. Synthetic Data Generation Pipeline ---
np.random.seed(42)
num_samples = 5000

data = {
    "duration": np.random.exponential(scale=2, size=num_samples),
    "protocol_type": np.random.choice(["tcp", "udp", "icmp"], size=num_samples),
    "service": np.random.choice(["http", "ftp", "smtp", "private"], size=num_samples),
    "flag": np.random.choice(["SF", "S0", "REJ"], size=num_samples),
    "src_bytes": np.random.randint(0, 50000, size=num_samples),
    "dst_bytes": np.random.randint(0, 50000, size=num_samples),
    "count": np.random.randint(1, 500, size=num_samples),
    "srv_count": np.random.randint(1, 500, size=num_samples),
    "dst_host_count": np.random.randint(1, 255, size=num_samples),
    "dst_host_srv_count": np.random.randint(1, 255, size=num_samples),
}

df = pd.DataFrame(data)
df['label'] = 0
attack_condition = (df['count'] > 350) & (df['duration'] < 0.5) | (df['src_bytes'] > 45000)
df.loc[attack_condition, 'label'] = 1

# --- 2. Fit Preprocessing Pipeline Operators ---
categorical_cols = ['protocol_type', 'service', 'flag']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop(columns=['label'])
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Model Training ---
model = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)
print("🎉 Model successfully optimized for live deployment.")

# --- 4. Core Prediction Engine Function ---
def predict_network_status(duration, protocol, service, flag, src_bytes, dst_bytes, count, srv_count, dst_host_count, dst_host_srv_count):
    # Map raw user values into encoded metrics
    encoded_protocol = encoders['protocol_type'].transform([protocol])[0]
    encoded_service = encoders['service'].transform([service])[0]
    encoded_flag = encoders['flag'].transform([flag])[0]
    
    # Structure features identically to training columns
    input_features = np.array([[
        duration, encoded_protocol, encoded_service, encoded_flag, 
        src_bytes, dst_bytes, count, srv_count, dst_host_count, dst_host_srv_count
    ]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Determine the status and safe display metrics
    if prediction == 1:
        status = "❌ MALICIOUS THREAT DETECTED"
        color = "red"
        details = f"Warning: This traffic mimics a signature DoS attack or payload exfiltration loop. (Confidence: {probabilities[1]*100:.1f}%)"
    else:
        status = "✅ CLEAN TRAFFIC PASSED"
        color = "green"
        details = f"Normal operational packet behavior verified. (Confidence: {probabilities[0]*100:.1f}%)"
        
    return status, details

# --- 5. Interactive UI Layout Design ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown(
        """
        # 🛡️ Real-Time Machine Learning Intrusion Detection System
        Use this interactive control dashboard to fine-tune network flow metrics and test how the internal Random Forest classification engine screens active threats.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Network Flow Parameters")
            duration = gr.Slider(minimum=0.0, maximum=10.0, value=1.5, label="Connection Duration (Seconds)")
            protocol = gr.Dropdown(choices=["tcp", "udp", "icmp"], value="tcp", label="Protocol Type")
            service = gr.Dropdown(choices=["http", "ftp", "smtp", "private"], value="http", label="Network Service")
            flag = gr.Dropdown(choices=["SF", "S0", "REJ"], value="SF", label="Connection Status Flag")
            src_bytes = gr.Slider(minimum=0, maximum=60000, value=1200, label="Source Payload (Bytes)")
            dst_bytes = gr.Slider(minimum=0, maximum=60000, value=3400, label="Destination Payload (Bytes)")
            count = gr.Slider(minimum=1, maximum=500, value=25, label="Connections to Same Host (Last 2 Sec)")
            srv_count = gr.Slider(minimum=1, maximum=500, value=20, label="Connections to Same Service (Last 2 Sec)")
            dst_host_count = gr.Slider(minimum=1, maximum=255, value=50, label="Destination Host Connection Counter")
            dst_host_srv_count = gr.Slider(minimum=1, maximum=255, value=80, label="Destination Host Service Connection Counter")
            
            submit_btn = gr.Button("Analyze Traffic Flow", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Live Firewall Verdict")
            output_verdict = gr.Textbox(label="Security Status Verdict", placeholder="Awaiting analytics computation...")
            output_details = gr.Textbox(label="Confidence Diagnostics", placeholder="Submit traffic metrics to evaluate threat telemetry.")
            
    submit_btn.click(
        fn=predict_network_status,
        inputs=[duration, protocol, service, flag, src_bytes, dst_bytes, count, srv_count, dst_host_count, dst_host_srv_count],
        outputs=[output_verdict, output_details]
    )

# Launch system inside the web container env
demo.launch(server_name="0.0.0.0", server_port=7860)
