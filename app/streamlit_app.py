"""Streamlit Web Application for Weed vs Crop Discrimination.

Allows users to upload field images, adjust confidence thresholds, visualize plant detections,
and inspect plant counts and weed pressure metrics.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported when launched via streamlit run
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image
import streamlit as st
import torch

from src.inference.predict import draw_detections, load_model, predict_image, summarize

st.set_page_config(
    page_title="Weed vs Crop AI Detector",
    page_icon="🌿",
    layout="wide",
)

CHECKPOINT_DEFAULT = "models/best.pt"


@st.cache_resource
def get_detector(checkpoint_path: str, device: str):
    """Cache loaded detector model across sessions."""
    model, _ = load_model(checkpoint_path, device=device)
    return model


def main():
    st.title("🌿 Field-Image Weed vs Crop AI Detector")
    st.markdown(
        "Detects, localizes, and discriminates **crops** and **weeds** in real agricultural field photographs "
        "using a fine-tuned Faster R-CNN MobileNetV3-Large FPN architecture."
    )

    st.sidebar.header("Model Settings")
    checkpoint_file = st.sidebar.text_input("Checkpoint Path", value=CHECKPOINT_DEFAULT)
    score_threshold = st.sidebar.slider(
        "Confidence Score Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Detections below this threshold are omitted to reduce false alarms.",
    )

    device_option = "cuda" if torch.cuda.is_available() else "cpu"
    device_choice = st.sidebar.selectbox("Device", options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], index=0)

    # Checkpoint validation
    ckpt_path = Path(checkpoint_file)
    if not ckpt_path.is_file():
        st.error(
            f"⚠️ Model checkpoint not found at `{checkpoint_file}`. "
            f"Please train the model first or verify the checkpoint path. Mock predictions are strictly forbidden."
        )
        st.info("Run `python -m src.training.train --config configs/laptop.yaml` to generate a checkpoint.")
        return

    try:
        model = get_detector(str(ckpt_path), device=device_choice)
    except Exception as e:
        st.error(f"Error loading model from {ckpt_path}: {e}")
        return

    # Image upload
    uploaded_file = st.file_uploader("Upload Field Image (JPG / PNG)", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not read uploaded image: {e}")
            return

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        if st.button("Detect Plants", type="primary"):
            with st.spinner("Analyzing field image..."):
                detections = predict_image(model, image, score_threshold=score_threshold, device=device_choice)
                summary = summarize(detections)
                annotated_img = draw_detections(image, detections)

            with col2:
                st.subheader("Detection Results")
                st.image(annotated_img, use_container_width=True)

            # Metric KPIs
            st.markdown("### 📊 Field Analysis Summary")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Plants", f"{summary['total_plants']}")
            kpi2.metric("Crop Plants", f"{summary['n_crop']}", delta="Preserve")
            kpi3.metric("Weed Plants", f"{summary['n_weed']}", delta="Target", delta_color="inverse")
            kpi4.metric("Weed Pressure", f"{summary['weed_share'] * 100:.1f}%")

            # Warning conditions
            if summary["total_plants"] == 0:
                st.warning("⚠️ No plant detections met the confidence threshold. Check image lighting or adjust threshold.")
            elif summary["mean_confidence"] < 0.60:
                st.warning(
                    f"⚠️ Low average confidence ({summary['mean_confidence']:.2f}). "
                    f"Results may be unreliable for this scene (possible domain shift or unusual lighting)."
                )

            # Detailed detections table
            if detections:
                st.markdown("### 📋 Detected Instances Table")
                df = pd.DataFrame(detections)[["label", "score", "box"]]
                df["box"] = df["box"].apply(lambda b: f"[{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}]")
                st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.caption(
        "**Research Prototype Disclaimer**: This system is designed for research, educational, and scouting experimentation. "
        "It is not certified for autonomous pesticide or herbicide application without human field oversight."
    )


if __name__ == "__main__":
    main()
