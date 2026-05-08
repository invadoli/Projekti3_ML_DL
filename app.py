import streamlit as st
import torch
from PIL import Image
import numpy as np
from sod_model import SODModel
from data_loader import get_transforms

st.set_page_config(
    page_title="SOD Saliency Detection Demo",
    layout="wide"
)

st.title("Saliency Object Detection")
st.write("Live Demo")


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SODModel()

    try:
        state_dict = torch.load(
            'best_sod_model.pth',
            map_location=device
        )

        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)

    except Exception as e:
        st.warning(
            "Weight file 'best_sod_model.pth' not found. Using untrained weights."
        )
        st.error(f"Error: {e}")

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()


st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)


uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    transform = get_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

        mask = prediction.cpu().squeeze().numpy()

        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        binary_mask = (mask > threshold).astype(np.uint8)

    img_array = np.array(image.resize((224, 224)))

    overlay = img_array.copy()

    overlay[binary_mask == 1] = [255, 0, 0]

    combined = (
        img_array * 0.7 + overlay * 0.3
    ).astype(np.uint8)

    resized_image = image.resize((244, 244))

    mask_display = Image.fromarray(
        (mask * 255).astype(np.uint8)
    ).resize((244, 244))

    combined_display = Image.fromarray(
        combined
    ).resize((244, 244))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")

        st.image(
            resized_image,
            width=244
        )

    with col2:
        st.subheader("Saliency Mask")

        st.image(
            mask_display,
            width=244
        )

    with col3:
        st.subheader("Final Detection")

        st.image(
            combined_display,
            width=244
        )

    st.success("Object successfully detected!")