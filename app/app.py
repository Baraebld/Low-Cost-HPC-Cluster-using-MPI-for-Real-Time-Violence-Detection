# app/app.py
import sys
from pathlib import Path
import logging
import streamlit as st
import torch
import tempfile
import traceback
from src.video_utils import load_video_frames
from src.load_model import load_trained_model, get_model_info

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Project root for imports
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="🔥 Violence Detection in Videos",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Violence Detection Model**

    This app uses a hybrid deep learning model combining 3D-CNN and Transformer architectures for video violence classification.
    Upload a video to evaluate prediction probabilities.

    ---
    **Model Details:**
    - Backbone: 3D ResNet-18 (`r3d_18`)
    - Temporal Encoder: Transformer (3 layers, 8 heads)
    - Input: 8 frames (224x224 RGB, ImageNet normalized)
    - Optimal Threshold: **0.65** (Precision: 97.25%, Recall: 93.75%)
    """)

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.65,
        step=0.05,
        help="Empirically tuned optimal threshold from experimental results (default: 0.65)"
    )

    # Model info will be populated after loading
    model_info_placeholder = st.empty()

# =========================
# Main App
# =========================
st.title("🔥 Violence Detection in Videos")
st.markdown(
    """
    Upload a video and get a **real-time prediction** if it contains violent content.
    The model analyzes video frames and provides probability scores.
    """
)

# =========================
# Load model with error handling
# =========================
@st.cache_resource
def load_model_cached():
    """Cache the model loading to avoid reloading on every rerun."""
    try:
        with st.spinner("Loading AI model... This may take a moment."):
            model, device = load_trained_model()
            model_info = get_model_info(model)
            logger.info(f"Model loaded successfully: {model_info}")
            return model, device, model_info
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

model, device, model_info = load_model_cached()

# Update sidebar with model info
with model_info_placeholder:
    st.markdown("**Model Information:**")
    st.json({
        "Parameters": f"{model_info['total_parameters']:,}",
        "Device": device.upper(),
        "Architecture": model_info['model_type']
    })

# =========================
# Video Upload Section
# =========================
st.header("📹 Upload Video")
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Supported formats: MP4, AVI, MOV, MKV, WebM. Max file size: 200MB",
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Validate file size (200MB limit)
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
    if file_size > 200:
        st.error(f"❌ File too large: {file_size:.1f}MB. Maximum allowed: 200MB")
        st.stop()

    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.getvalue())
        video_path = tfile.name

    try:
        # Display video and prediction side by side
        with st.container():
            col1, col2 = st.columns([1.3, 1])

            with col1:
                st.subheader("🎬 Video Preview")
                st.video(video_path)

            with col2:
                st.subheader("🔍 Analysis Results")
                with st.spinner("Analyzing video frames..."):
                    try:
                        # Load and preprocess video frames
                        video_tensor = load_video_frames(video_path).unsqueeze(0).to(device)

                        # Make prediction
                        with torch.no_grad():
                            output = model(video_tensor)
                            prob = torch.sigmoid(output).item()
                            violence_prob = prob * 100
                            non_violence_prob = 100 - violence_prob

                        # Display results
                        st.metric(
                            "Violence Probability",
                            f"{violence_prob:.1f}%",
                            delta=f"{'High' if violence_prob > 70 else 'Low' if violence_prob < 30 else 'Medium'}"
                        )

                        # Progress bars
                        st.progress(violence_prob / 100, text="Violence")
                        st.progress(non_violence_prob / 100, text="Non-Violence")

                        # Main prediction with color coding based on threshold
                        if prob >= threshold:
                            st.error(f"⚠️ **POTENTIAL VIOLENCE DETECTED** (Score: {violence_prob:.1f}% ≥ Threshold: {threshold*100:.0f}%)")
                            st.markdown("**Recommendation:** Flag for manual security review")
                        else:
                            st.success(f"✅ **NO VIOLENCE DETECTED** (Score: {violence_prob:.1f}% < Threshold: {threshold*100:.0f}%)")
                            st.markdown("**Status:** Content classified as safe")

                        # Additional details
                        with st.expander("📊 Detailed Analysis"):
                            st.write(f"**Raw Model Probability:** {prob:.4f}")
                            st.write(f"**Applied Threshold:** {threshold:.2f}")
                            st.write(f"**Processing Device:** {device.upper()}")

                    except Exception as e:
                        logger.error(f"Error during prediction: {e}")
                        st.error(f"❌ Error analyzing video: {str(e)}")
                        if st.checkbox("Show technical details"):
                            st.code(traceback.format_exc())

    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        st.error(f"❌ Error processing video file: {str(e)}")

    finally:
        # Cleanup temporary file
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not cleanup temp file: {e}")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        🤖 <strong>HPC Video Violence Detection</strong> | PyTorch & Streamlit |
        Academic Project (2025–2026, ESI SBA) |
        Developed by <a href='https://github.com/Baraebld' target='_blank'><strong>El Baraa BELAID</strong></a>
    </div>
    """,
    unsafe_allow_html=True
)