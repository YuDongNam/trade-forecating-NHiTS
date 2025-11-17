"""
Import 방식으로 파이프라인을 실행하는 스크립트
Colab 등에서 tqdm 출력이 더 깔끔하게 보입니다.

사용법:
    python run_pipeline.py
    
또는 Python 코드에서:
    from run_pipeline import run
    run()
"""
import sys
from pathlib import Path

def run(config_dir: str = "config", **kwargs):
    """
    Import 방식으로 파이프라인 실행
    
    Args:
        config_dir: 설정 디렉토리 경로
        **kwargs: 추가 인자 (val_mode, val_tail_months 등)
    """
    from src.pipeline.run_all import main
    
    # sys.argv 설정
    args = ['run_pipeline.py', '--config_dir', config_dir]
    
    if 'val_mode' in kwargs:
        args.extend(['--val_mode', kwargs['val_mode']])
    if 'val_tail_months' in kwargs:
        args.extend(['--val_tail_months', str(kwargs['val_tail_months'])])
    if 'val_start' in kwargs:
        args.extend(['--val_start', kwargs['val_start']])
    if 'val_end' in kwargs:
        args.extend(['--val_end', kwargs['val_end']])
    if 'skip_training' in kwargs and kwargs['skip_training']:
        args.append('--skip_training')
    if 'skip_evaluation' in kwargs and kwargs['skip_evaluation']:
        args.append('--skip_evaluation')
    if 'skip_forecast' in kwargs and kwargs['skip_forecast']:
        args.append('--skip_forecast')
    
    sys.argv = args
    main()


if __name__ == "__main__":
    # 기본 실행
    run()
    
    # 또는 인자와 함께:
    # run(config_dir="config", val_mode="range", val_start="2023-01-01", val_end="2023-12-01")
