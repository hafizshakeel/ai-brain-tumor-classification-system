import streamlit as st
import yaml
import os
import time
from predict import Predictor
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
import pandas as pd

# =====================
# Page Setup
# =====================
st.set_page_config(
    page_title="NeuroDiagnost AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for medical styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;500;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #005b96;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #455a64;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .prediction-card {
        background: linear-gradient(to right, #f8f9fa, #f1f5f9);
        border-radius: 12px;
        padding: 25px;
        border-left: 5px solid #005b96;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #005b96;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #455a64;
        font-weight: 400;
    }
    
    .high-confidence {color: #00796b !important;}
    .medium-confidence {color: #ff9800 !important;}
    .low-confidence {color: #f44336 !important;}
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #005b96 !important;
        color: white !important;
    }
    
    .upload-box {
        border: 2px dashed #005b96;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #455a64;
        font-size: 0.9rem;
        font-weight: 300;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 5px;
    }
    .badge-success {background-color: #E8F5E9; color: #2E7D32;}
    .badge-warning {background-color: #FFF8E1; color: #F57F17;}
    .badge-danger {background-color: #FFEBEE; color: #C62828;}
    .badge-info {background-color: #E3F2FD; color: #1565C0;}
    
    /* Medical Info Box */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 0.9rem;
    }
    
    /* Report styling */
    .report-header {
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .report-section {
        margin-bottom: 15px;
    }
    
    .report-label {
        font-weight: 500;
        color: #455a64;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #005b96;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# =====================
# Load Config
# =====================
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_data_dir = config["training"]["train_data_dir"]

# =====================
# Sidebar
# =====================
st.sidebar.markdown("<div class='sidebar-header'>🔬 NeuroDiagnost AI</div>", unsafe_allow_html=True)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["model.pth", "final_model.pth"],
    index=1
)
model_path = os.path.join(
    "artifacts/training",
    model_choice
)

# Professional Settings
st.sidebar.markdown("### Settings")
theme = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
show_advanced = st.sidebar.checkbox("Show Advanced Information", value=False)

# Information Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### About Brain Tumors")
tumor_info = {
    "Glioma": "Originates in the glial cells that surround and support neurons. Grade varies from I (least aggressive) to IV (most aggressive).",
    "Meningioma": "Forms in the meninges, the membranes surrounding the brain and spinal cord. Usually benign but can cause symptoms due to pressure.",
    "Pituitary": "Develops in the pituitary gland at the base of the brain. Can affect hormone production.",
}

for tumor, desc in tumor_info.items():
    with st.sidebar.expander(f"{tumor} Tumor"):
        st.write(desc)

st.sidebar.markdown("---")
st.sidebar.caption("💡 This tool is for educational purposes only and not intended for clinical diagnosis.")

# =====================
# Predictor
# =====================
@st.cache_resource
def load_predictor(model_path):
    return Predictor(model_path, train_data_dir)

predictor = load_predictor(model_path)

# =====================
# Session State (for history)
# =====================
if "history" not in st.session_state:
    st.session_state.history = []
    
if "patient_id" not in st.session_state:
    st.session_state.patient_id = f"PT-{datetime.datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"

# =====================
# Helper Functions
# =====================
def get_confidence_color(confidence):
    if confidence >= 0.85:
        return "high-confidence"
    elif confidence >= 0.65:
        return "medium-confidence"
    else:
        return "low-confidence"

def get_confidence_badge(confidence):
    if confidence >= 0.85:
        return "badge badge-success"
    elif confidence >= 0.65:
        return "badge badge-warning"
    else:
        return "badge badge-danger"

def get_confidence_message(confidence):
    if confidence >= 0.85:
        return "High confidence prediction. Results are likely reliable."
    elif confidence >= 0.65:
        return "Moderate confidence. Consider further review."
    else:
        return "Low confidence. Results should be interpreted with caution."

# =====================
# Main Layout
# =====================
st.markdown("<div class='main-title'>NeuroDiagnost AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced AI-powered brain tumor classification from MRI scans</div>", unsafe_allow_html=True)

# Two-column layout for patient info and upload
col_info, col_upload = st.columns([1, 2])

with col_info:
    st.markdown("### 📋 Patient Information")
    patient_id = st.text_input("Patient ID", value=st.session_state.patient_id, disabled=True)
    scan_date = st.date_input("Scan Date", value=datetime.datetime.now())
    scan_type = st.selectbox("MRI Sequence", ["T1-weighted", "T2-weighted", "FLAIR", "Contrast-Enhanced T1"])

with col_upload:
    st.markdown("### 🔍 Upload MRI Scan")
    
    # Create a more attractive upload area
    upload_col1, upload_col2 = st.columns([3, 2])
    
    with upload_col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop MRI scan here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with upload_col2:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("**Important:**")
        st.markdown("• Upload T1-weighted contrast enhanced MRI scan")
        st.markdown("• Ensure image is clear and properly cropped")
        st.markdown("• DICOM format not supported (convert to JPEG/PNG first)")
        st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    # Display a loading spinner
    with st.spinner("Analyzing MRI scan..."):
        # Save temp
        temp_path = "temp_uploaded_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Add some artificial delay to simulate processing
        time.sleep(0.5)
        
        # Run prediction
        result = predictor.predict(temp_path)
        
        # Add to history
        st.session_state.history.append({
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_id": patient_id,
            "filename": uploaded_file.name,
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "probs": result.get("probs", {}),
            "scan_type": scan_type,
            "scan_date": scan_date.strftime("%Y-%m-%d")
        })
    
    # Show success message
    st.success("Analysis complete!")
    
    # =====================
    # Tabs for Dashboard
    # =====================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Diagnostic Report", "🧠 MRI Analysis", "📑 Technical Data", "📈 History"])
    
    # --- Tab 1: Diagnostic Report ---
    with tab1:
        st.markdown("<h3 style='text-align:center;'>Brain MRI Analysis Report</h3>", unsafe_allow_html=True)
        
        st.markdown("<div class='report-header'>", unsafe_allow_html=True)
        report_col1, report_col2 = st.columns(2)
        
        with report_col1:
            st.markdown("<span class='report-label'>Patient ID:</span> " + patient_id, unsafe_allow_html=True)
            st.markdown("<span class='report-label'>Scan Type:</span> " + scan_type, unsafe_allow_html=True)
        
        with report_col2:
            st.markdown("<span class='report-label'>Scan Date:</span> " + scan_date.strftime("%Y-%m-%d"), unsafe_allow_html=True)
            st.markdown("<span class='report-label'>Analysis Date:</span> " + datetime.datetime.now().strftime("%Y-%m-%d"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Finding Section
        st.markdown("#### Findings")
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        confidence_class = get_confidence_color(result['confidence'])
        badge_class = get_confidence_badge(result['confidence'])
        
        st.markdown(f"<h3>Primary Finding: <span class='{confidence_class}'>{result['prediction']}</span> <span class='{badge_class}'>{result['confidence']:.1%} confidence</span></h3>", unsafe_allow_html=True)
        
        st.markdown("<p>" + get_confidence_message(result['confidence']) + "</p>", unsafe_allow_html=True)
        
        # Clinical Correlation Note
        st.markdown("#### Impression")
        if result['prediction'] == "No Tumor":
            st.markdown("No significant abnormalities detected in the MRI scan. Clinical correlation is recommended.")
        else:
            st.markdown(f"Findings are consistent with **{result['prediction']}**. Further clinical correlation and potential follow-up imaging is recommended.")
            
            # Additional medical information based on tumor type
            if "Glioma" in result['prediction']:
                st.markdown("*Gliomas are typically evaluated based on WHO grading system. Consider additional sequences and potential biopsy for definitive grading.*")
            elif "Meningioma" in result['prediction']:
                st.markdown("*Meningiomas are typically slow-growing and benign. Assessment of dural attachment and vascular supply may be beneficial.*")
            elif "Pituitary" in result['prediction']:
                st.markdown("*Pituitary tumors may affect hormone production. Endocrinological evaluation is recommended. Consider dedicated pituitary protocol MRI.*")
        
        # Disclaimer
        st.markdown("#### Disclaimer")
        st.markdown("*This analysis was generated using artificial intelligence and should be reviewed by a qualified medical professional. This tool is not FDA approved for clinical diagnosis.*")
        
        # Export options
        st.markdown("#### Report Actions")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "💾 Download Report (PDF)",
                "This would generate a PDF report in a real application",
                file_name=f"NeuroDiagnost_Report_{patient_id}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        with export_col2:
            st.button("📧 Send to Referring Physician")

    # --- Tab 2: MRI Analysis ---
    with tab2:
        analysis_col1, analysis_col2 = st.columns([1, 1.2])
        
        with analysis_col1:
            st.image(uploaded_file, caption="Uploaded MRI", use_container_width=True)
            
            # Image metadata
            with st.expander("Image Metadata"):
                img = Image.open(uploaded_file)
                st.write(f"Format: {img.format}")
                st.write(f"Size: {img.size}")
                st.write(f"Mode: {img.mode}")
        
        with analysis_col2:
            # Confidence visualization
            st.markdown("#### Confidence Assessment")
            
            # Create three metric cards in a row
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value {get_confidence_color(result['confidence'])}'>{result['confidence']:.1%}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Confidence Score</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Add other metrics
            with metric_col2:
                certainty = 1.0 - (sum(result["probs"].values()) - max(result["probs"].values()))
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value {get_confidence_color(certainty)}'>{certainty:.1%}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Class Separation</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with metric_col3:
                clinical_threshold = result['confidence'] > confidence_threshold
                threshold_color = "high-confidence" if clinical_threshold else "low-confidence"
                threshold_text = "Above Threshold" if clinical_threshold else "Below Threshold"
                
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value {threshold_color}'>{threshold_text}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Clinical Threshold</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Probability Distribution visualization
            st.markdown("#### Diagnostic Probability Distribution")
            if "probs" in result:
                probs = result["probs"]
                # Convert to DataFrame for better visualization
                df = pd.DataFrame({
                    "Tumor Type": list(probs.keys()),
                    "Probability": list(probs.values())
                }).sort_values("Probability", ascending=False)
                
                # Create a better looking bar chart
                prob_fig = px.bar(
                    df,
                    x="Tumor Type",
                    y="Probability",
                    color="Probability",
                    color_continuous_scale=["#f8bbd0", "#c5cae9", "#90caf9", "#1976d2"],
                    text=df["Probability"].apply(lambda p: f"{p:.1%}"),
                    height=300
                )
                
                prob_fig.update_layout(
                    plot_bgcolor="white",
                    xaxis_title="",
                    yaxis_title="",
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                
                prob_fig.update_traces(
                    textposition='outside',
                    textfont=dict(
                        size=12,
                        color="black"
                    )
                )
                
                st.plotly_chart(prob_fig, use_container_width=True)
                
                # Display a table of exact values
                st.markdown("#### Detailed Probability Scores")
                prob_df = pd.DataFrame({
                    "Tumor Type": list(probs.keys()),
                    "Probability": [f"{p:.2%}" for p in probs.values()]
                }).sort_values("Tumor Type")
                
                st.table(prob_df)
            

    # --- Tab 3: Technical Data ---
    with tab3:
        tech_col1, tech_col2 = st.columns([1, 1])
        
        with tech_col1:
            st.markdown("#### Model Information")
            st.json({
                "Model": model_choice,
                "Architecture": "Swin Transformer (swin_tiny_patch4_window7_224)",
                "Framework": "PyTorch",
                "Input Resolution": "224x224",
                "Classes": 4,
                "Preprocessing": "Normalization, Resizing",
                "Version": "1.0"
            })
            
            st.markdown("#### Raw Prediction Data")
            st.json(result)
        
        with tech_col2:
            st.markdown("#### Performance Metrics")
            
            # Confidence Gauge - more professional looking
            conf_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result["confidence"]*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Level", 'font': {'size': 24}},
                delta={'reference': confidence_threshold*100, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#1976d2"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#ffcdd2'},
                        {'range': [50, 75], 'color': '#ffecb3'},
                        {'range': [75, 100], 'color': '#c8e6c9'},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': confidence_threshold*100
                    }
                }
            ))
            
            conf_fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            st.plotly_chart(conf_fig, use_container_width=True)
            
            if show_advanced:
                st.markdown("#### Algorithmic Decision Process")
                st.markdown("""
                1. **Preprocessing**: Input image resized to 224x224 and normalized
                2. **Feature Extraction**: Swin Transformer extracts hierarchical features
                3. **Classification**: Linear layer produces logits for 4 classes
                4. **Softmax**: Converts logits to probability distribution
                5. **Decision**: Class with highest probability selected
                """)
                
                st.markdown("#### Potential Limitations")
                st.markdown("""
                - Limited to 4 tumor categories
                - Performance depends on image quality
                - May not generalize to different MRI acquisition parameters
                - Not intended for clinical diagnosis
                - No segmentation of tumor boundaries
                """)

    # --- Tab 4: History ---
    with tab4:
        st.markdown("#### Patient Scan History")
        
        if len(st.session_state.history) == 0:
            st.info("No history yet. Upload an image to start tracking.")
        else:
            history_df = pd.DataFrame(st.session_state.history)
            
            # Format columns for display
            if not history_df.empty:
                history_df["confidence"] = history_df["confidence"].apply(lambda x: f"{x:.1%}")
                
                # Select and rename columns for display
                display_df = history_df[["time", "patient_id", "scan_type", "prediction", "confidence"]].copy()
                display_df.columns = ["Timestamp", "Patient ID", "Scan Type", "Finding", "Confidence"]
                
                st.dataframe(display_df, use_container_width=True)
            
            # Allow history clearing
            if st.button("Clear History"):
                st.session_state.history = []
                st.experimental_rerun()

# =====================
# Footer
# =====================
st.markdown("<div class='footer'>NeuroDiagnost AI | Advanced Brain Tumor Analysis Tool | Version 2.0<br>© 2025 NeuroDiagnost Medical Technologies. Not for clinical use.</div>", unsafe_allow_html=True)
