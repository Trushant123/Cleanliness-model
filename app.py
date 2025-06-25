import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model
@st.cache_resource
def load_classifier():
    return load_model("mobilenetv2_clean_dirty_classifier.h5")

model = load_classifier()
classes = ['Clean', 'Dirty']

# Title
st.title("Railway/Street Cleanliness Classifier")
st.markdown("Upload an image and the model will classify it as **Clean** or **Dirty** and show **why**.")

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Grad-CAM utilities
def generate_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img

# Process and predict
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224)) / 255.0
    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    img_array = np.expand_dims(image_resized, axis=0)
    prediction_prob = model.predict(img_array)[0][0]
    predicted_index = int(prediction_prob > 0.5)
    predicted_label = classes[predicted_index]

    st.success(f"### Prediction: {predicted_label}")
    st.markdown(f"**Confidence:** `{prediction_prob * 100:.2f}%`")

    # Confidence bar graph
    probs = [1 - prediction_prob, prediction_prob]
    fig, ax = plt.subplots()
    ax.barh(classes, probs, color=['green', 'red'])
    ax.set_xlim([0, 1])
    ax.set_xlabel('Confidence')
    ax.set_title('Prediction Confidence')
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v * 100:.2f}%", va='center', fontweight='bold')
    st.pyplot(fig)

    # Grad-CAM
    st.markdown("### Grad-CAM Visualization")
    heatmap = generate_gradcam_heatmap(img_array, model)
    gradcam_result = overlay_heatmap(image_rgb, heatmap)
    st.image(gradcam_result, caption="Highlighted regions influencing the prediction", use_column_width=True)
