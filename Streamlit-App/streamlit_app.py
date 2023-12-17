import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Model
from PIL import Image
import IPython.display as display


# Required Functions
def load_image(image_path, max_dim):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img.thumbnail([max_dim, max_dim])
    img = np.array(img, dtype=np.uint8)
    img = np.expand_dims(img, axis=0)
    return img


def deprocess_inception_image(img):
    img = 255 * (img + 1) / 2
    return np.array(img, np.uint8)


def array_to_img(array, deprocessing=False):
    if deprocessing:
        array = deprocess_inception_image(array)

    if np.ndim(array) > 3:
        assert array.shape[0] == 1
        array = array[0]

    return Image.fromarray(array)


def show_image(img):
    image = array_to_img(img)
    display.display(image)


def deep_dream_model(model, layer_names):
    model.trainable = False
    outputs = [model.get_layer(name).output for name in layer_names]
    new_model = Model(inputs=model.input, outputs=outputs)
    return new_model


def get_loss(activations):
    loss = []
    for activation in activations:
        loss.append(tf.math.reduce_mean(activation))
    return tf.reduce_sum(loss)


def model_output(model, inputs):
    return model(inputs)


def get_loss_and_gradient(model, inputs, total_variation_weight=0):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        activations = model_output(model, inputs)
        loss = get_loss(activations)
        loss = loss + total_variation_weight * tf.image.total_variation(inputs)
    grads = tape.gradient(loss, inputs)
    grads /= tf.math.reduce_std(grads) + 1e-8
    return loss, grads


def run_gradient_ascent(model, inputs, progress_bar, epochs=1, steps_per_epoch=1, weight=0.05,
                        total_variation_weight=0):
    img = tf.convert_to_tensor(inputs)
    for i in range(epochs):
        for step in range(steps_per_epoch):
            _, grads = get_loss_and_gradient(model, img, total_variation_weight)
            img = img + grads * weight
            img = tf.clip_by_value(img, -1.0, 1.0)

            # Update progress bar
            progress_value = ((i * steps_per_epoch) + step) / (epochs * steps_per_epoch)
            progress_bar.progress(progress_value)

    return img.numpy(), progress_bar


centered_text = """
        <div style="text-align: center;">
            Built with ❤️ by Unnati
        </div>
        """

# Streamlit App
st.title("Deep Dream Streamlit App")
st.write("Upload an image to generate mesmerizing Deep Dream images. Adjust the parameters in the sidebar to get "
         "different effects.")
st.write("Image generation may take a while depending on the parameters chosen, kindly be patient.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Checkboxes for selecting layers
st.sidebar.title("Adjust parameters for different effects!")
layer_checkboxes = []
for i in range(1, 11):
    default_value = (i == 5)  # Set default to True for layer 5, False for others
    layer_checkbox = st.sidebar.checkbox(f"Layer {i}", value=default_value)
    layer_checkboxes.append(layer_checkbox)

# Sliders for parameter adjustments
epochs = st.sidebar.slider("Epochs", 1, 5, 2, help="Number of training epochs")
steps_per_epoch = st.sidebar.slider("Steps per Epoch", 1, 100, 50, help="Number of steps per epoch")
weight = st.sidebar.slider("Weight", 0.01, 0.1, 0.02, step=0.01, help="Weight for gradient ascent")
dim = st.sidebar.slider("Image Size", 128, 1024, 300,
                        help='Choose maximum dimension of image. Higher size image will take longer to be generated.')

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    input_image = load_image(uploaded_file, max_dim=dim)
    preprocessed_image = inception_v3.preprocess_input(input_image)

    # Create Inception model and modify for deep dream
    inception = inception_v3.InceptionV3(weights="imagenet", include_top=False)

    # Select layers based on user input
    selected_layers = [f'mixed{i}' for i, checkbox in enumerate(layer_checkboxes, start=1) if checkbox]
    dream_model = deep_dream_model(inception, selected_layers)

    progress_bar = st.progress(0.0)  # Initialize progress bar
    # Run gradient ascent
    (image_array, progress_bar) = run_gradient_ascent(dream_model, preprocessed_image, progress_bar, epochs=epochs,
                                                      steps_per_epoch=steps_per_epoch, weight=weight)

    # Convert numpy arrays to PIL images
    dream_pil_image = array_to_img(deprocess_inception_image(image_array))

    # Display the Deep Dream image
    st.image(dream_pil_image, caption='Deep Dream Image', width=400)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(centered_text, unsafe_allow_html=True)
