# Google Colab 실행 가이드

이 가이드는 NHiTS 프로젝트를 Google Colab에서 실행하는 방법을 설명합니다.

## 방법 1: 노트북 사용 (권장)

1. **노트북 열기**
   - `colab_setup.ipynb` 파일을 Colab에 업로드하거나
   - Colab에서 새 노트북을 만들고 아래 단계를 따라하세요

2. **저장소 클론 또는 파일 업로드**
   ```python
   # GitHub에 업로드한 경우
   !git clone <your-repo-url>
   %cd <repo-name>
   
   # 또는 직접 파일 업로드
   from google.colab import files
   uploaded = files.upload()
   ```

3. **데이터 파일 업로드**
   - Colab의 파일 업로드 기능 사용
   - 또는 Google Drive에 업로드 후 마운트

4. **패키지 설치**
   ```python
   !pip install -r requirements.txt
   ```

5. **실행**
   ```python
   !python -m src.pipeline.run_all
   ```

## 방법 2: Google Drive 연동

### 1. Google Drive에 프로젝트 업로드

1. 프로젝트 전체를 zip으로 압축
2. Google Drive에 업로드
3. Colab에서 Drive 마운트:

```python
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 디렉토리로 이동
%cd /content/drive/MyDrive/NHiTS
```

### 2. 데이터 파일 확인

```python
import os
from pathlib import Path

# 데이터 파일 확인
data_dir = Path("data")
csv_files = list(data_dir.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {f.name}")
```

### 3. 실행

```python
# 패키지 설치
!pip install neuralforecast torch pandas numpy scikit-learn matplotlib pydantic tqdm

# 파이프라인 실행
!python -m src.pipeline.run_all
```

## 방법 3: 직접 파일 업로드

### 단계별 가이드

1. **Colab 새 노트북 생성**

2. **디렉토리 구조 생성**
   ```python
   import os
   os.makedirs("data", exist_ok=True)
   os.makedirs("src/config", exist_ok=True)
   os.makedirs("src/data", exist_ok=True)
   os.makedirs("src/model", exist_ok=True)
   os.makedirs("src/pipeline", exist_ok=True)
   os.makedirs("models", exist_ok=True)
   os.makedirs("results", exist_ok=True)
   ```

3. **파일 업로드**
   ```python
   from google.colab import files
   import shutil
   
   # 소스 코드 파일들 업로드
   uploaded = files.upload()
   
   # CSV 파일은 data/ 폴더로 이동
   for filename in uploaded.keys():
       if filename.endswith('.csv'):
           shutil.move(filename, f"data/{filename}")
   ```

4. **패키지 설치 및 실행**
   ```python
   !pip install neuralforecast torch pandas numpy scikit-learn matplotlib pydantic tqdm
   !python -m src.pipeline.run_all
   ```

## 주의사항

### GPU 사용
- Colab은 무료로 GPU를 제공합니다
- 런타임 > 런타임 유형 변경 > GPU 선택
- PyTorch는 자동으로 GPU를 감지합니다

### 세션 시간 제한
- 무료 Colab은 약 12시간 후 세션이 종료됩니다
- 장시간 학습이 필요한 경우:
  - 중간 체크포인트 저장
  - Google Drive에 모델 저장
  - 또는 Colab Pro 사용

### 메모리 부족 시
- 배치 크기 줄이기 (`src/config/base_config.py`에서 `batch_size` 수정)
- 한 번에 하나의 타겟만 학습
- `max_steps` 줄이기

## 빠른 시작 (한 셀에 모두)

```python
# 1. 디렉토리 생성
import os
from pathlib import Path
os.makedirs("data", exist_ok=True)
os.makedirs("src/config", exist_ok=True)
os.makedirs("src/data", exist_ok=True)
os.makedirs("src/model", exist_ok=True)
os.makedirs("src/pipeline", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2. 패키지 설치
!pip install neuralforecast torch pandas numpy scikit-learn matplotlib pydantic tqdm

# 3. 파일 업로드 (수동으로 파일 선택)
from google.colab import files
uploaded = files.upload()

# 4. 파일 정리
import shutil
for filename in uploaded.keys():
    if filename.endswith('.csv'):
        shutil.move(filename, f"data/{filename}")
    elif filename.startswith('src/'):
        # src 폴더 파일은 그대로 유지
        pass

# 5. 실행
!python -m src.pipeline.run_all
```

## 결과 다운로드

```python
from google.colab import files
import zipfile

# results 폴더 압축
with zipfile.ZipFile('results.zip', 'w') as zipf:
    for root, dirs, files_list in os.walk('results'):
        for file in files_list:
            zipf.write(os.path.join(root, file))

# 다운로드
files.download('results.zip')
```

## 문제 해결

### ImportError 발생 시
```python
import sys
sys.path.append('/content')  # 또는 프로젝트 루트 경로
```

### 경로 문제
- Colab은 `/content/`를 기본 작업 디렉토리로 사용
- `src/config/paths.py`는 자동으로 프로젝트 루트를 찾습니다

### CUDA out of memory
- `batch_size`를 줄이세요 (기본값: 32 → 16 또는 8)
- `max_steps`를 줄이세요 (기본값: 1500 → 500)

