# Import 방식 실행 가이드

Python 모듈을 import 방식으로 실행하는 방법입니다. Colab 환경에서 tqdm 출력이 더 깔끔하게 보입니다.

## 기본 방법

### 1. 전체 파이프라인 실행

**방법 A: Python 인터랙티브 셸 또는 스크립트**
```python
from src.pipeline.run_all import main
main()
```

**방법 B: 인자 전달 (argparse 사용)**
```python
import sys
from src.pipeline.run_all import main

# sys.argv 설정 (기본값 사용)
sys.argv = ['run_all.py', '--config_dir', 'config']
main()
```

**방법 C: 직접 함수 호출 (인자 없이)**
```python
from src.pipeline.run_all import main
from pathlib import Path

# config 디렉토리 기본값 사용
main()
```

### 2. 개별 단계 실행

**학습만 실행:**
```python
from src.pipeline.train import main as train_main
train_main()
```

**평가만 실행:**
```python
from src.pipeline.evaluate import main as evaluate_main
evaluate_main()
```

**예측만 실행:**
```python
from src.pipeline.forecast import main as forecast_main
forecast_main()
```

### 3. Colab 노트북에서 사용 (tqdm 출력 개선, 권장)

```python
# tqdm notebook 모드로 설정 (Colab에서 더 깔끔하게 보임)
from tqdm.notebook import tqdm

# 방법 A: run_pipeline.py 사용 (Jupyter 커널 인자 문제 해결)
from run_pipeline import run
run()

# 방법 B: 직접 import (parse_known_args로 Jupyter 인자 무시)
from src.pipeline.run_all import main as run_pipeline
run_pipeline()
```

### 4. 인자와 함께 실행

**검증 모드 오버라이드:**
```python
import sys
from src.pipeline.train import main as train_main

# CLI 인자 설정
sys.argv = [
    'train.py',
    '--config_dir', 'config',
    '--val_mode', 'range',
    '--val_start', '2023-01-01',
    '--val_end', '2023-12-01'
]

train_main()
```

**또는 직접 함수 호출 (함수 수정 필요):**
```python
from src.pipeline.train import train_target
from src.config.yaml_loader import (
    load_paths_config,
    load_train_config,
    load_validation_config,
    load_exogenous_config,
    load_targets_config,
)
from pathlib import Path

config_dir = Path("config")
paths_config = load_paths_config(config_dir)
train_config = load_train_config(config_dir)
validation_config = load_validation_config(config_dir)
exog_config = load_exogenous_config(config_dir)
targets_config = load_targets_config(config_dir)

# 특정 타겟만 학습
train_target(
    "Korea_Import",
    train_config,
    validation_config,
    exog_config,
    paths_config,
    compute_full_r2=True
)
```

## 비교: 모듈 실행 vs Import 실행

### 모듈 실행 방식 (기존)
```bash
python -m src.pipeline.run_all --config_dir config
```

**장점:**
- 명령줄에서 직접 실행 가능
- 인자 전달이 간단
- 배치 스크립트에 적합

**단점:**
- Colab에서 tqdm 출력이 깔끔하지 않을 수 있음

### Import 실행 방식 (권장 - Colab)
```python
from src.pipeline.run_all import main
main()
```

**장점:**
- Colab에서 tqdm 출력이 더 깔끔함
- Python 코드 내에서 직접 제어 가능
- 디버깅이 쉬움

**단점:**
- 인자 전달을 위해 sys.argv 조작 필요

## 실제 사용 예시

### 예시 1: Colab 노트북 셀
```python
# 셀 1: 설정
from tqdm.notebook import tqdm
from src.pipeline.run_all import main as run_pipeline

# 셀 2: 실행
run_pipeline()
```

### 예시 2: Python 스크립트
```python
# run_pipeline.py
from src.pipeline.run_all import main

if __name__ == "__main__":
    main()
```

실행:
```bash
python run_pipeline.py
```

### 예시 3: Jupyter/IPython
```python
%load_ext autoreload
%autoreload 2

from src.pipeline.run_all import main
main()
```

## 주의사항

1. **경로 설정**: 프로젝트 루트에서 실행해야 합니다
2. **인자 전달**: `sys.argv`를 조작하거나 함수를 직접 수정해야 합니다
3. **Colab 환경**: `tqdm.notebook`을 사용하면 출력이 더 깔끔합니다

