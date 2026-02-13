# VAE Decoder Layer-wise Analysis Tool
나는서재윤 
Stable Diffusion VAE의 디코더 레이어별로 두 이미지 간의 차이를 분석하는 도구입니다.
dㅇㅇㅇ아ㅓㅏ어ㅏ어ㅏ어
## 기능

- 두 이미지를 VAE에 입력하여 디코더의 각 레이어 출력 수집
- 레이어별 출력 차이 측정 (L2 distance, Cosine similarity)
- 차이가 큰 레이어 자동 식별
- 단일 이미지 쌍 분석 및 폴더 배치 처리 지원

## 설치

```bash
cd /home/jaey00ns/MedCLIP-SAMv2-main/vae
pip install torch torchvision numpy matplotlib seaborn pillow tqdm omegaconf
```

## 디렉토리 구조

```
vae/
├── analyze_vae_layers.py    # 메인 분석 스크립트
├── run_analysis.sh           # 실행 스크립트
├── README.md                 # 이 파일
└── outputs/                  # 출력 결과 (자동 생성)
    ├── single/              # 단일 이미지 쌍 결과
    │   ├── layer_differences.json
    │   ├── layer_differences_plot.png
    │   └── reconstructed_images.png
    └── batch/               # 폴더 배치 처리 결과
        ├── image_name_1/
        ├── image_name_2/
        └── aggregated_differences.png
```

## 사용법

### 1. 단일 이미지 쌍 분석

두 개의 이미지를 직접 지정하여 분석합니다.

```bash
# 셸 스크립트 실행 권한 부여
chmod +x run_analysis.sh

# L2 distance만 사용
./run_analysis.sh single --img1 tumor.jpg --img2 masked_tumor.jpg --method l2

# Cosine similarity만 사용
./run_analysis.sh single --img1 tumor.jpg --img2 masked_tumor.jpg --method cosine

# 둘 다 사용 (권장)
./run_analysis.sh single --img1 tumor.jpg --img2 masked_tumor.jpg --method both
```

또는 Python 직접 실행:

```bash
python analyze_vae_layers.py \
    --mode single \
    --img1 /path/to/image1.jpg \
    --img2 /path/to/image2.jpg \
    --method both \
    --device cuda:5 \
    --output ./outputs
```

### 2. 폴더 배치 처리

폴더 구조:
```
data/ultrasound_pairs/
├── positive/          # 종양 있는 이미지들
│   ├── case001.jpg
│   ├── case002.jpg
│   └── ...
└── negative/          # 종양 마스킹한 이미지들
    ├── case001.jpg    # positive와 같은 이름
    ├── case002.jpg
    └── ...
```

실행:

```bash
# 셸 스크립트 사용
./run_analysis.sh folder --folder ./data/ultrasound_pairs --method both

# 또는 Python 직접 실행
python analyze_vae_layers.py \
    --mode folder \
    --folder ./data/ultrasound_pairs \
    --method both \
    --device cuda:5 \
    --output ./outputs
```

## 출력 설명

### 단일 이미지 쌍 분석 결과

1. **layer_differences.json**: 각 레이어별 수치 결과
   ```json
   {
     "decoder.conv_in": {
       "l2_value": 0.0234,
       "cosine_value": 0.0012,
       "shape": [1, 512, 64, 64]
     },
     ...
   }
   ```

2. **layer_differences_plot.png**: 레이어별 차이 시각화
   - 막대 그래프로 각 레이어의 차이 표시
   - 상위 5개 레이어 하이라이트

3. **reconstructed_images.png**: 원본 및 복원 이미지 비교
   - 좌측: 원본 이미지
   - 우측: VAE 복원 이미지

### 폴더 배치 처리 결과

- 각 이미지 쌍마다 개별 폴더에 위와 동일한 결과 저장
- **aggregated_differences.png**: 모든 이미지 쌍의 평균 차이
- **aggregated_results.json**: 평균 및 표준편차

## 차이 측정 방법

### L2 Distance (Euclidean Distance)
- 두 텐서 간의 유클리드 거리 측정
- 값이 클수록 차이가 큼
- 절대적인 픽셀 값 차이에 민감

### Cosine Similarity
- 두 벡터의 방향 유사도 측정 (0~1)
- Cosine Dissimilarity = 1 - Cosine Similarity
- 크기보다는 패턴 차이에 민감

### Both (권장)
- 두 방법을 모두 계산하여 종합적으로 분석

## GPU 설정

기본값은 `cuda:5`입니다. 변경하려면:

```bash
# 셸 스크립트 수정
vim run_analysis.sh
# DEVICE="cuda:5" → DEVICE="cuda:0" 등으로 변경

# 또는 실행 시 지정
./run_analysis.sh single --img1 a.jpg --img2 b.jpg --device cuda:0
```

## 예제 시나리오: 유방암 초음파

### 시나리오
- 종양이 있는 초음파 이미지 (positive)
- 종양을 마스킹한 초음파 이미지 (negative)
- **목표**: VAE 디코더가 종양의 유무를 어느 레이어에서 가장 크게 인식하는가?

### 분석 절차

1. 이미지 준비
```bash
mkdir -p data/breast_ultrasound/positive
mkdir -p data/breast_ultrasound/negative
# 이미지 복사
```

2. 단일 쌍 테스트
```bash
./run_analysis.sh single \
    --img1 data/breast_ultrasound/positive/sample.jpg \
    --img2 data/breast_ultrasound/negative/sample.jpg \
    --method both
```

3. 결과 확인
```bash
# 결과 확인
ls outputs/single/
# layer_differences.json, layer_differences_plot.png, reconstructed_images.png

# 상위 레이어 확인 (터미널 출력에서)
# TOP 5 Layers with Largest Differences 참고
```

4. 전체 데이터 분석
```bash
./run_analysis.sh folder \
    --folder data/breast_ultrasound \
    --method both

# 집계 결과 확인
open outputs/batch/aggregated_differences.png
```

### 해석

- **초기 레이어** (conv_in, mid blocks)에서 차이가 크면:
  → VAE가 저수준 특징(texture, edge)에서 차이 인식

- **중간 레이어** (up blocks)에서 차이가 크면:
  → 의미적 특징 레벨에서 차이 인식

- **후기 레이어** (conv_out)에서 차이가 크면:
  → 최종 이미지 재구성 단계에서 차이 발생

## 문제 해결

### 모델 체크포인트 없음
```bash
# 모델 다운로드 필요
cd ../stable-diffusion
# Hugging Face 등에서 Stable Diffusion v1.5 모델 다운로드
```

### CUDA out of memory
- GPU 메모리 부족 시 더 작은 배치나 CPU 사용
```bash
./run_analysis.sh single --img1 a.jpg --img2 b.jpg --device cpu
```

### 이미지 크기 문제
- 자동으로 512×512로 리사이즈되므로 모든 크기 지원

## 추가 커스터마이징

`analyze_vae_layers.py`를 직접 수정하여:
- 다른 레이어 추가/제거 (`_register_hooks` 메서드)
- 다른 차이 측정 방법 추가 (`compute_difference` 메서드)
- 시각화 스타일 변경 (`visualize_differences` 메서드)

## 참고

- 이 도구는 Stable Diffusion의 VAE를 사용합니다
- 의료 영상 분석에 특화된 도구는 아니므로 연구 목적으로만 사용하세요
- 실제 임상 진단에는 사용하지 마세요
