
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
# Import the predictor class from our generated script
from inference import FabricPredictor

st.set_page_config(page_title="Fabric Defect Detector", page_icon="🧵")

st.title("🧵 Fabric Defect Detection System")
st.markdown("**Custom Ensemble Model** | *No Scikit-Learn Dependencies*")

@st.cache_resource
def load_model():
    return FabricPredictor('deployment_model.pkl')

try:
    model = load_model()
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Uploaded Fabric', use_column_width=True)

    with st.spinner('Analyzing...'):
        label, confidence, all_probs = model.predict(image)

    with col2:
        st.subheader(f"Result: {label}")
        st.metric(label="Confidence", value=f"{confidence:.2%}")

        # Prepare chart data
        # Ensure we display probabilities in the correct order of classes
        sorted_indices = np.argsort(list(model.idx_to_class.keys()))
        class_names = [model.idx_to_class[i] for i in sorted_indices]

        probs_df = pd.DataFrame({
            'Class': class_names,
            'Probability': all_probs
        })
        st.bar_chart(probs_df.set_index('Class'))
