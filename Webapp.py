import streamlit as st
import pandas as pd
import joblib

# ตั้งชื่อเว็บ
st.set_page_config(page_title="Smart Irrigation Predictor", layout="centered")
st.title("💧 Smart Irrigation Prediction Web App")

st.markdown("""
อัปโหลดไฟล์ .CSV ที่มีข้อมูลจากเซ็นเซอร์ แล้วระบบจะทำนายว่าแต่ละแถวควรรดน้ำหรือไม่ 🌱

**Status 1 = รดน้ำ**  |  **Status 0 = ไม่รดน้ำ**
""")

# โหลดโมเดล
@st.cache_resource
def load_model():
    return joblib.load("Smart_irrigation_model.pkl")

model = load_model()

# รับไฟล์ CSV จากผู้ใช้
uploaded_file = st.file_uploader("📥 เลือกไฟล์ CSV สำหรับทำนาย:", type="csv")

if uploaded_file:
    try:
        # โหลดข้อมูล
        input_df = pd.read_csv(uploaded_file)
        st.success("✅ อัปโหลดไฟล์เรียบร้อยแล้ว!")

        # ตรวจสอบว่ามีคอลัมน์ 'Status' หรือไม่ และลบทิ้ง
        if 'Status' in input_df.columns:
            input_df = input_df.drop('Status', axis=1)

        # ทำนายผล
        predictions = model.predict(input_df)

        # รวมผลลัพธ์
        result_df = input_df.copy()
        result_df['Predicted Status'] = predictions
        result_df['Predicted Label'] = result_df['Predicted Status'].apply(lambda x: '💧 รดน้ำ' if x == 1 else '☀️ ไม่รดน้ำ')

        # แสดงผลลัพธ์ในตาราง
        st.subheader("🔍 ผลการทำนาย")
        st.dataframe(result_df, use_container_width=True)

        # ดาวน์โหลดผลลัพธ์
        csv_download = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 ดาวน์โหลดผลลัพธ์เป็น CSV", data=csv_download, file_name="prediction_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลไฟล์: {e}")
