import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import torch.nn.functional as F

# ==========================================
# 0. 페이지 설정 및 디바이스 지정
# ==========================================
st.set_page_config(page_title="영수증 디노이징 및 OCR 파이프라인", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 모델 및 OCR 엔진 로드 (캐싱을 통해 속도 향상)
# ==========================================
@st.cache_resource
def load_ocr_engine():
    # 최신 PaddleOCR 설정 (MKLDNN 충돌 방지 포함)
    return PaddleOCR(use_textline_orientation=True, lang='korean', enable_mkldnn=False)

# [사용자 정의 U-Net 모델 구조 - 학습 시 사용한 구조와 동일]
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetDenoising(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetDenoising, self).__init__()
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512 + 256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256 + 128, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128 + 64, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64 + 64, 64)

        self.outc = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)
        return logits

@st.cache_resource
def load_denoise_model():
    model = UNetDenoising(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("model/unet_receipt_model.pth", map_location=device))
    model.eval()
    return model

ocr = load_ocr_engine()

try:
    model = load_denoise_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error_msg = str(e)

# ==========================================
# 2. 웹 UI 구성
# ==========================================
st.title("🧾 고해상도 영수증 디노이징 & OCR 웹 서비스")
st.write("노이즈가 낀 영수증 이미지를 업로드하면, U-Net으로 선명하게 복원한 뒤 PaddleOCR로 텍스트를 추출합니다.")

if not model_loaded:
    st.warning(f"⚠️ U-Net 가중치 파일(`unet_receipt_model.pth`)을 찾을 수 없거나 로드에 실패했습니다. (에러: {load_error_msg})")

uploaded_file = st.file_uploader("영수증 이미지 업로드 (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 이미지 읽기
    image = Image.open(uploaded_file).convert("L") # 그레이스케일 변환
    
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
        st.info(f"💡 이미지 해상도가 너무 높아 메모리 보호를 위해 최대 크기({max_size}px)로 자동 리사이즈되었습니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 업로드된 원본 (Noisy Input)")
        st.image(image, use_container_width=True)
        
    # 이미지 전처리 및 텐서 변환
    img_np = np.array(image).astype(np.float32) / 255.0
    input_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, H, W]
    
    # U-Net 추론 (디노이징)
    if model_loaded:
        with torch.no_grad():
            pred_tensor = model(input_tensor)
            
        # 텐서를 OpenCV 이미지로 변환
        pred_np = pred_tensor.squeeze(0).squeeze(0).cpu().numpy()
        pred_img_np = (pred_np * 255).astype(np.uint8)
        pred_pil = Image.fromarray(pred_img_np)
        
        with col2:
            st.subheader("2. U-Net 복원 결과 (Model Output)")
            st.image(pred_pil, use_container_width=True)
    else:
        pred_img_np = (img_np * 255).astype(np.uint8)

    # OCR 추출 함수
    def extract_texts(img_gray):
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        result = ocr.predict(img_bgr)
        extracted = []
        if result:
            for res in result:
                if isinstance(res, dict) and 'rec_texts' in res and 'rec_scores' in res:
                    for t, s in zip(res['rec_texts'], res['rec_scores']):
                        if t.strip():
                            extracted.append((t, s))
        return extracted

    st.markdown("---")
    st.subheader("📊 OCR 텍스트 추출 및 신뢰도 비교")
    
    with st.spinner("OCR 분석 중입니다..."):
        noisy_texts = extract_texts(np.array(image))
        pred_texts = extract_texts(pred_img_np) if model_loaded else []

    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown("**[Noisy Input 인식 결과]**")
        if noisy_texts:
            for t, s in noisy_texts:
                st.write(f"- `{t}` (신뢰도: {s:.2f})")
        else:
            st.info("인식된 텍스트가 없습니다.")
            
    with res_col2:
        st.markdown("**[U-Net 복원본 인식 결과]**")
        if pred_texts:
            for t, s in pred_texts:
                st.write(f"- `{t}` (신뢰도: {s:.2f})")
        else:
            st.info("인식된 텍스트가 없습니다.")