import streamlit as st
import whisper
import os
import tempfile

# Cấu hình trang Streamlit
st.set_page_config(page_title="Video to Text Converter", page_icon="🎥", layout="centered")

st.title("🎥 Chuyển đổi Video/Audio thành Văn bản")
st.write("Ứng dụng sử dụng mô hình OpenAI Whisper để tự động nhận dạng giọng nói.")

# Hàm tải mô hình Whisper và lưu vào bộ nhớ cache để tránh tải lại nhiều lần
@st.cache_resource
def load_model():
    # Các tùy chọn model: 'tiny', 'base', 'small', 'medium', 'large'
    # 'base' phù hợp cho ứng dụng chạy mượt mà, không tốn quá nhiều RAM/CPU
    return whisper.load_model("base")

with st.spinner("Đang tải mô hình Whisper (vui lòng đợi trong giây lát)..."):
    model = load_model()

# Trình tải tệp tin (Đã sửa lỗi st.file_file_uploader thành st.file_uploader)
uploaded_file = st.file_uploader(
    "Tải lên tệp video hoặc âm thanh của bạn", 
    type=["mp4", "mkv", "avi", "mov", "mp3", "wav", "m4a"]
)

if uploaded_file is not None:
    # Hiển thị trình phát file tùy thuộc vào định dạng
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension in ["mp4", "mkv", "avi", "mov"]:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)
    
    # Nút bấm bắt đầu xử lý
    if st.button("Bắt đầu chuyển chữ"):
        try:
            with st.spinner("Đang xử lý và nhận diện giọng nói..."):
                # Ghi tệp tạm thời vì Whisper yêu cầu đường dẫn tệp trực tiếp
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Tiến hành nhận diện văn bản
                result = model.transcribe(tmp_file_path)
                text_output = result["text"]

                # Xóa tệp tạm sau khi xử lý xong
                os.remove(tmp_file_path)

            # Hiển thị kết quả
            st.success("Chuyển đổi thành công!")
            st.subheader("Kết quả văn bản:")
            st.write(text_output)

            # Nút tải xuống kết quả dưới dạng file .txt
            st.download_button(
                label="Tải xuống tệp văn bản (.txt)",
                data=text_output,
                file_name=f"{uploaded_file.name.split('.')[0]}_transcript.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
