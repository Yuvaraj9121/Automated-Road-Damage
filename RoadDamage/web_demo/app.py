import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">🛣️ Automated Road Damage Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered road damage detection using YOLO</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    Upload a road image and the AI model will automatically identify
    visible road damage and display the detected regions.
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("RoadDamage/model/best.pt")

model = load_model()

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "📷 Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("AI Detection")

        if st.button("🔍 Detect Road Damage", use_container_width=True):

            with st.spinner("Analyzing road image..."):

                results = model.predict(
                    source=np.array(image),
                    conf=0.10
                )

            result = results[0]

            # -----------------------------
            # Display annotated image
            # -----------------------------
            annotated_image = result.plot()

            st.image(
                annotated_image,
                channels="BGR",
                use_container_width=True
            )

            # -----------------------------
            # Detection information
            # -----------------------------
            if result.boxes is not None and len(result.boxes) > 0:

                st.success(
                    f"Detected {len(result.boxes)} road damage object(s)"
                )

                st.subheader("Detection Details")

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    st.write(
                        f"**{class_name}** — "
                        f"Confidence: **{confidence:.1%}**"
                    )

            else:

                st.warning(
                    "No road damage detected. "
                    "Try another road image or an image with clearer damage."
                )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "Automated Road Damage Detection | "
    "YOLO-based Computer Vision Project"
)