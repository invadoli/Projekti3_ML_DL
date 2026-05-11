import streamlit as st
import torch
import numpy as np
from PIL import Image
from sod_model import SODModel
from torchvision import transforms
import time

st.set_page_config(page_title="Salient Object Detection Demo", layout="wide")
st.title("Salient Object Detection (SOD) Demo")
st.write("Upload an image to see the AI identify the most 'salient' object.")

@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SODModel()
    try:
        state_dict = torch.load('best_sod_model.pth', map_location=device)
        model.load_state_dict(state_dict)
    except:
        st.error("Model weights not found! Make sure 'best_sod_model.pth' is in the folder.")
    model.to(device)
    model.eval()
    return model, device

model, device = load_trained_model()

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = preprocess(image)
    
    start_time = time.time()
    with torch.no_grad():
        prediction = model(input_tensor)
    inference_time = time.time() - start_time
    
    pred_mask = prediction.squeeze().cpu().numpy()
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    
    with col2:
        st.image(pred_mask_img, caption="Predicted Saliency Mask", use_container_width=True)

    mask_resized = pred_mask_img.resize(image.size, resample=Image.BILINEAR)
    mask_array = np.array(mask_resized) / 255.0
    
    img_array = np.array(image) / 255.0
    overlay = img_array.copy()
    overlay[:, :, 0] = overlay[:, :, 0] * (1 - mask_array * 0.5) + (mask_array * 0.5) 
    
    with col3:
        st.image(overlay, caption=f"Overlay (Inf: {inference_time:.3f}s)", use_container_width=True)
    
    st.success(f"Inference completed in {inference_time:.4f} seconds!")
else:
    st.info("Please upload an image in the sidebar to start.")