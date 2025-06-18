import streamlit as st
import pandas as pd
import joblib

# Setup page
st.set_page_config(page_title="Prediabeta", layout="wide")

# Custom style untuk sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: #a2d2ff;
        }
        button[kind="primary"] {
            width: 100% !important;
            text-align: left !important;
            margin-bottom: 10px;
            border-radius: 10px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
rf_model = joblib.load("random_forest_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_features = joblib.load("feature_names.pkl")

# Dictionary mapping
ya_tidak_dict = {"Ya": 1, "Tidak": 0}
gender_dict = {"Perempuan": 0, "Laki-laki": 1}
genhlth_dict = {
    "Sangat baik": 1,
    "Baik": 2,
    "Cukup Baik": 3,
    "Kurang Baik": 4,
    "Tidak Baik": 5
}

# Sidebar navigasi
with st.sidebar:
    st.title("Prediabetix")
    if st.button("🏠 Home"):
        st.session_state.page = "home"
    if st.button("🔍 Prediksi Risiko Diabetes"):
        st.session_state.page = "prediksi"

# Default halaman
if "page" not in st.session_state:
    st.session_state.page = "home"

# Halaman HOME
if st.session_state.page == "home":
    st.title("🔬 Selamat Datang di Aplikasi Prediabeta")
    st.markdown("## 🧪 Cek Risiko Diabetes Anda Secara Mudah dan Cepat")

    st.image("hero.jpeg", use_container_width=True, caption="Ilustrasi Pemeriksaan Diabetes")

    st.markdown("""
    <div style='text-align: justify; font-size:16px;'>
        Aplikasi <b>Prediabeta</b> adalah alat bantu prediksi yang dirancang untuk membantu Anda
        mengenali potensi risiko diabetes berdasarkan indikator kesehatan pribadi seperti tekanan darah,
        kebiasaan merokok, aktivitas fisik, dan lainnya.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Fitur Utama")
    fitur_col1, fitur_col2 = st.columns(2)

    with fitur_col1:
        st.success("✅ Prediksi Risiko Diabetes dengan AI")
        st.info("📊 Analisis data kesehatan Anda secara akurat.")

    with fitur_col2:
        st.warning("📝 Formulir interaktif & mudah diisi")
        st.error("🔒 Data Anda aman dan tidak dibagikan ke pihak ketiga.")

    st.markdown("---")
    st.markdown("### 🧭 Cara Menggunakan")
    st.markdown("""
    1. Klik menu **Prediksi Risiko Diabetes** di sidebar.
    2. Isi form berdasarkan kondisi kesehatan Anda.
    3. Klik tombol **Prediksi Sekarang**.
    4. Hasil prediksi akan langsung ditampilkan.
    """)

    st.markdown("---")
    st.markdown("### 🩺 Kategori Risiko")
    st.markdown("""
    - 🟢 **Non Diabetes** – Risiko sangat rendah  
    - 🟡 **Pre-Diabetes** – Perlu perhatian  
    - 🔴 **Diabetes** – Segera konsultasi ke tenaga medis
    """)

    st.markdown("""
    <br><br>
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Aplikasi ini hanya bersifat prediktif, bukan diagnosis medis.<br>
        Konsultasikan ke dokter untuk pemeriksaan lebih lanjut.
    </div>
    """, unsafe_allow_html=True)

# Halaman PREDIKSI
elif st.session_state.page == "prediksi":
    st.title("🔍 Form Prediksi Risiko Diabetes")

    def radio_q(label):
        return st.radio(label, list(ya_tidak_dict.keys()), index=None, horizontal=True)

    data = {
        "HighBP": radio_q("Apakah memiliki tekanan darah tinggi?"),
        "HighChol": radio_q("Apakah menderita kolesterol tinggi?"),
        "CholCheck": radio_q("Apakah sudah cek kolesterol dalam 5 tahun terakhir?"),
        "BMI": st.text_input("Masukkan BMI Anda (contoh: 25.0)", value=""),
        "Smoker": radio_q("Apakah Anda perokok aktif?"),
        "Stroke": radio_q("Apakah Anda pernah mengalami stroke?"),
        "HeartDiseaseorAttack": radio_q("Apakah Anda memiliki riwayat penyakit jantung?"),
        "PhysActivity": radio_q("Apakah Anda berolahraga dalam 30 hari terakhir?"),
        "Fruits": radio_q("Apakah Anda rutin mengonsumsi buah?"),
        "Veggies": radio_q("Apakah Anda rutin mengonsumsi sayur?"),
        "HvyAlcoholConsump": radio_q("Apakah Anda mengonsumsi alkohol berat?"),
        "AnyHealthcare": radio_q("Apakah Anda memiliki akses layanan kesehatan?"),
        "GenHlth": st.radio("Bagaimana kondisi kesehatan umum Anda?", list(genhlth_dict.keys()), index=None, horizontal=True),
        "MentHlth": st.slider("Jumlah hari gangguan mental (0-30)", 0, 30, 0),
        "PhysHlth": st.slider("Jumlah hari gangguan fisik (0-30)", 0, 30, 0),
        "DiffWalk": radio_q("Apakah Anda kesulitan berjalan?"),
        "Sex": st.radio("Jenis Kelamin", list(gender_dict.keys()), index=None, horizontal=True),
        "Age": st.number_input("Usia Anda", min_value=1, max_value=100, value=25)
    }

    def map_prediction(pred):
        return {
            0: "🟢 Non Diabetes",
            1: "🟡 Pre-Diabetes",
            2: "🔴 Diabetes"
        }.get(pred, "Tidak diketahui")

    if st.button("🔎 Prediksi Sekarang"):
        try:
            missing = [k for k, v in data.items() if v in (None, "")]
            if missing:
                st.warning("⚠ Lengkapi semua input sebelum prediksi.")
                st.write("Field belum diisi:", missing)
            else:
                input_data = {
                    "HighBP": ya_tidak_dict[data["HighBP"]],
                    "HighChol": ya_tidak_dict[data["HighChol"]],
                    "CholCheck": ya_tidak_dict[data["CholCheck"]],
                    "BMI": float(data["BMI"]),
                    "Smoker": ya_tidak_dict[data["Smoker"]],
                    "Stroke": ya_tidak_dict[data["Stroke"]],
                    "HeartDiseaseorAttack": ya_tidak_dict[data["HeartDiseaseorAttack"]],
                    "PhysActivity": ya_tidak_dict[data["PhysActivity"]],
                    "Fruits": ya_tidak_dict[data["Fruits"]],
                    "Veggies": ya_tidak_dict[data["Veggies"]],
                    "HvyAlcoholConsump": ya_tidak_dict[data["HvyAlcoholConsump"]],
                    "AnyHealthcare": ya_tidak_dict[data["AnyHealthcare"]],
                    "GenHlth": genhlth_dict[data["GenHlth"]],
                    "MentHlth": data["MentHlth"],
                    "PhysHlth": data["PhysHlth"],
                    "DiffWalk": ya_tidak_dict[data["DiffWalk"]],
                    "Sex": gender_dict[data["Sex"]],
                    "Age": int(data["Age"])
                }

                df_input = pd.DataFrame([input_data])
                df_input = df_input[expected_features]
                scaled_input = scaler.transform(df_input)

                rf_pred = rf_model.predict(scaled_input)[0]
                dt_pred = dt_model.predict(scaled_input)[0]

                st.subheader("📊 Hasil Prediksi")
                st.write("🌲 Random Forest:", map_prediction(rf_pred))
                st.write("🌳 Decision Tree:", map_prediction(dt_pred))

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
