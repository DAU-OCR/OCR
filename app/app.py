import streamlit as st
from logic.ocr_engine import run_ocr
from logic.image_filter import select_best_image
from utils.excel_writer import save_to_excel
from PIL import Image

# 상단바 streamlit 설정에서 테마 Light로 변경 필요
st.set_page_config(page_title="불법주차 OCR 처리기", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #2c3e50; font-size: 40px; font-weight: bold; margin-bottom: 0.5em;'>
        불법주차 차량 OCR 자동 분석 시스템
    </h1>
    """,
    unsafe_allow_html=True
)

# ====== 옵션 설정 (좌측 사이드바) ======
with st.sidebar:
    st.markdown("### ⚙️ 옵션 설정")

    with st.expander("1. 실행 디바이스 선택"):
        device_choice = st.radio(
            "디바이스를 선택하세요:",
            options=["자동", "CPU", "GPU"],
            index=0
        )

        st.markdown(f"✅ 현재 선택된 디바이스: **{device_choice}**")
        if device_choice == "자동":
            st.info("자동 모드에서는 향후 CPU/GPU 성능을 비교해 빠른 쪽을 자동 선택할 예정입니다. (미구현)")

    with st.expander("2. OCR 정확도 향상 옵션 (미구현)"):
        st.text("향후 적용 예정: 전처리, 다중모델 등")

    with st.expander("3. 엑셀 저장 포맷 설정 (미구현)"):
        st.text("저장 형식, 컬럼 선택 등 사용자 지정 옵션 예정")

    with st.expander("4. 장애인 구역 인식 기능 (미구현)"):
        st.text("딥러닝 기반 표지 인식 기능 도입 예정")

# 파일 업로드
uploaded_files = st.file_uploader("이미지를 업로드하세요 (여러 장 가능)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("1️⃣ 업로드된 이미지 미리보기")
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        images.append((uploaded_file.name, image))
        st.image(image, caption=uploaded_file.name, use_column_width=True)

    st.subheader("2️⃣ 차량번호 인식 중...")

    test_license_numbers = [
    "12가3456", "23나4567", "34다5678", "45라6789", "56마7890",
    "67바8901", "78사9012", "89아0123", "90자1234", "01차2345"
    ]

    # 1차 OCR 수행
    ocr_results = []
    for i, (filename, image) in enumerate(images):
        # 테스트용 번호 지정
        fake_number = test_license_numbers[i % len(test_license_numbers)]
        confidence = 0.99  # 가짜 confidence 값 (원하는 대로 조절 가능)

        ocr_results.append({
            "filename": filename,
            "image": image,
            "text": fake_number,
            "confidence": confidence
        })

    # 가장 좋은 이미지 선택
    best_result = select_best_image(ocr_results)

    if best_result:
        st.success(f"가장 선명한 번호판 이미지: {best_result['filename']}")
        st.image(best_result["image"], caption="선택된 이미지", use_column_width=True)
        st.write("🔍 인식된 번호:", best_result["text"])

        st.subheader("3️⃣ 엑셀 저장 준비 중...")

        # results 리스트 구성: 모든 이미지에 대해 2차 OCR까지 포함
        results = []
        for r in ocr_results:
            rerun_text, _ = run_ocr(r["image"])
            results.append({
                "filename": r["filename"],
                "image": r["image"],
                "text": r["text"],
                "rerun_text": rerun_text
            })

        # 엑셀 저장
        excel_bytes, file_name = save_to_excel(results)

        st.download_button(
            label="📥 엑셀 다운로드",
            data=excel_bytes,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("번호판을 인식할 수 없습니다. 더 선명한 사진을 업로드해 주세요.")