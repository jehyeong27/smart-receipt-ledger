# 영수증 디노이징 및 OCR 웹 서비스

노이즈가 있는 영수증 이미지를 업로드하면 U-Net 모델로 이미지를 복원하고 PaddleOCR로 원본과 복원본의 텍스트를 각각 인식하는 Streamlit 웹 앱입니다.

## 주요 기능

- PNG, JPG, JPEG 영수증 이미지 업로드
- 입력 이미지를 그레이스케일로 변환하고, 긴 변이 1024px를 넘으면 자동으로 축소
- 학습된 U-Net 모델을 사용한 이미지 디노이징
- 원본과 복원본의 OCR 결과 및 인식 신뢰도 비교
- CUDA 사용 가능 시 PyTorch 추론에 GPU 사용

## 프로젝트 구조

```text
.
├── app.py
├── requirements.txt
├── model/
│   ├── unet_receipt_model.pth
│   └── best_receipt_model.keras
└── .streamlit/
    └── config.toml
```

앱에서 사용하는 U-Net 가중치는 `model/unet_receipt_model.pth`입니다. 이 파일이 없거나 로드에 실패하면 앱은 경고를 표시하고 원본 이미지로 OCR을 수행합니다. `best_receipt_model.keras`는 현재 앱 코드에서 사용하지 않습니다.

## 설치 및 실행

1. Python 가상 환경을 만들고 활성화합니다.

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. PaddleOCR 실행에 필요한 PaddlePaddle을 사용 환경에 맞게 설치합니다. CPU 또는 CUDA 버전에 따라 설치 방법이 다를 수 있으므로 [PaddlePaddle 설치 안내](https://www.paddlepaddle.org.cn/install/quick)를 확인하세요.

3. 프로젝트 의존성을 설치합니다.

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. 프로젝트 루트에서 앱을 실행합니다.

   ```powershell
   streamlit run app.py
   ```

5. 브라우저에서 Streamlit이 출력한 로컬 주소를 열고 영수증 이미지를 업로드합니다.

## 참고 사항

- PaddleOCR 및 모델 초기화에는 첫 실행 시 시간이 걸릴 수 있습니다.
- PyTorch 모델은 `model/unet_receipt_model.pth`의 상대 경로에서 로드되므로 프로젝트 루트에서 실행해야 합니다.
- 입력 이미지는 메모리 사용을 줄이기 위해 긴 변 기준 최대 1024px로 축소됩니다.
