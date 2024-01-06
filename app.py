import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Memuat model yang telah dilatih
model_path = "model_efficient_pneumonia.h5"
loaded_model = tf.keras.models.load_model((model_path),
                                          custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

# Fungsi untuk memproses gambar input
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Aplikasi Streamlit
st.title("Pneumonia Detection")

# Mengunggah gambar melalui Streamlit
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    st.image(uploaded_file, caption="Gambar yang diunggah.", use_column_width=True)

    # Menambahkan tombol prediksi
    if st.button("Prediksi"):
        # Memproses gambar dan membuat prediksi
        img_array = preprocess_image(uploaded_file)
        predictions = loaded_model.predict(img_array)

        # Menampilkan prediksi
        st.subheader("Prediksi:")
        class_names = ["negative", "severe"]  # Ganti dengan nama kelas yang sesuai
        predicted_class = class_names[np.argmax(predictions)]

        st.write(f"Model memprediksi: {predicted_class}")