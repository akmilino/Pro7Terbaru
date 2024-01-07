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
selected_page = st.sidebar.radio("Menu", ["Home", "Prediction", "About Us"])

if selected_page == "Home":
    st.title("Selamat Datang di Web PROTECT")
    st.write("Web ini dapat anda gunakan untuk mengunggah gambar dan melakukan prediksi menggunakan AI.")
    st.image("Screenshot (1).png", use_column_width=True)

elif selected_page == "Prediction":
    st.title("Upload Gambar")

    # Mengunggah gambar melalui Streamlit
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah
        st.image(uploaded_file, caption="Gambar Berhasil Diunggah.", use_column_width=True)

        # Menambahkan tombol prediksi
        if st.button("Prediksi"):
            # Memproses gambar dan membuat prediksi
            img_array = preprocess_image(uploaded_file)
            predictions = loaded_model.predict(img_array)

            # Menampilkan prediksi
            st.subheader("Prediksi:")
            class_names = ["negative", "severe"]  # Ganti dengan nama kelas yang sesuai
            predicted_class = class_names[np.argmax(predictions)]

            st.write(f"Hasil Prediksi: {predicted_class}")

elif selected_page == "About Us":
    st.title("Tentang Kami")
    st.write("PRO7 adalah tim pengembang dalam bidang AI yang berkomitmen untuk terus mengembangkan aplikasi yang menggunakan AI.")
    st.write("pada aplikasi ini AI berperan dalam pendeteksi Pneumonia sehingga memudahkan untuk mengklasifikasikan gambar paru-paru penderita Pneumonia.")
    st.write("untuk website ini masih dalam pengembangan, jangan ragu untuk memberikan saran kontak kami pro7@gmail.com")