import streamlit as st
from inference_sdk import InferenceHTTPClient
import os
from PIL import Image

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="XMMpwFnxYxEd1SjnNPjU"
)

st.title("Sushrut AI")

uploaded_file = st.file_uploader("Upload an X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    temp_path = "temp_uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner("Processing..."):
        try:
            result = CLIENT.infer(temp_path, model_id="chest-x-ray-images-kheq6/2")
            
            predictions = result.get('predictions', [])
            if predictions:
                st.subheader("Results:")
                for pred in predictions:
                    class_name = pred.get('class', 'Unknown')
                    confidence = pred.get('confidence', 0)
                    confidence_pct = confidence * 100
                    st.success(f"**{class_name}**: {confidence_pct:.2f}% confidence")
            else:
                st.info("No class detected.")
        except Exception as e:
            st.error(f"Error during inference: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)