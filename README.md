# 🎬 Label Studio Helper

AI 기반 비디오 세그멘테이션 도구

## ✨ 특징

- 🎥 **자동 비디오 세그멘테이션**: ResNet 기반 AI로 정적 구간 자동 제거
- 🚀 **GPU 가속 지원**: NVIDIA GPU를 활용한 고속 처리
- 🌐 **gradio UI**: 웹 브라우저 기반 유려한 인터페이스
- 📦 **경량 설계**: ~50MB (PyTorch는 런타임 자동 설치)
- 🔒 **안전한 권한**: 일반 사용자 권한으로만 실행

## 🖥️ 시스템 요구사항

- Windows 10/11
- NVIDIA GPU (GPU 가속 사용 시)
- 10GB 이상의 디스크 공간

## 📥 설치 및 실행

### 방법 1: ZIP 다운로드

1. [Release 페이지](링크)에서 최신 버전 다운로드
2. ZIP 압축 해제
3. `label_studio_helper.exe` 실행

### 방법 2: 소스에서 빌드

```bash
# 의존성 설치
pip install -r requirements.txt

# 빌드
python build.py

# 실행
dist/label_studio_helper/label_studio_helper.exe
```

## 🚀 사용 방법

### 첫 실행

1. `label_studio_helper.exe` 실행
2. 브라우저가 자동으로 열립니다 (http://127.0.0.1:7860)
3. GPU 가속을 사용하려면 "PyTorch 설정" 탭에서 PyTorch 설치

### 비디오 세그멘테이션

1. "비디오 세그멘테이션" 탭 선택
2. 비디오 파일 업로드
3. 설정 조정:
   - **정적 임계값**: 0.97 (높을수록 더 많이 제거)
   - **최소 정적 길이**: 0.1초 (짧은 정적 구간 무시)
   - **목표 세그먼트 길이**: 30초
   - **GPU 가속**: 체크 (PyTorch 설치 필요)
4. "세그멘테이션 시작" 클릭

### 결과

- 세그먼트 비디오 파일: `result_seg/` 폴더
- 유사도 그래프: `similarity_graph.png`
- 메타데이터: `metadata.json`

## ⚙️ 설정

### PyTorch 설치

- "PyTorch 설정" 탭에서 자동 설치
- 최신 CUDA 버전 지원 (CUDA 13.0+)
- 설치 경로: `%APPDATA%/LabelStudioHelper/pytorch/`

### 데이터 위치

모든 데이터는 사용자별로 저장됩니다:
- PyTorch: `%APPDATA%/LabelStudioHelper/pytorch/`
- 로그: `%APPDATA%/LabelStudioHelper/logs/`

## ⚠️ 중요 사항

### 일반 사용자 권한으로 실행

**이 앱은 관리자 권한으로 실행할 수 없습니다.**

- ❌ 우클릭 → "관리자 권한으로 실행" 사용 금지
- ✅ 일반적으로 더블클릭하여 실행

**이유**: PyTorch 등 Add-on 설치 시 권한 충돌을 방지하기 위함입니다.

## 🛠️ 개발

### 프로젝트 구조

```
LabelStudioHelper/
├── app.py                    # gradio 메인 UI
├── core/
│   ├── video_segmenter.py   # 세그멘테이션 로직
│   ├── feature_extractor.py  # ResNet feature 추출
│   └── pytorch_installer.py  # PyTorch 자동 설치
├── utils/
├── hooks/                    # PyInstaller hooks
├── requirements.txt
├── build.py                  # 빌드 스크립트
└── label_studio_helper.spec  # PyInstaller spec
```

### 빌드

```bash
python build.py
```

## 📄 라이선스

MIT License

## 🤝 기여

이슈 및 PR 환영합니다!

## 📮 문의

- GitHub Issues: [링크]
- Email: [이메일]
