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
    # Jupyter/Colab 환경에서 sys.argv를 완전히 대체
    # 원본 sys.argv 백업
    original_argv = sys.argv.copy()
    
    try:
        # 새로운 sys.argv 설정 (Jupyter 커널 인자 제외)
        new_argv = ['run_pipeline.py', '--config_dir', config_dir]
        
        if 'val_mode' in kwargs:
            new_argv.extend(['--val_mode', kwargs['val_mode']])
        if 'val_tail_months' in kwargs:
            new_argv.extend(['--val_tail_months', str(kwargs['val_tail_months'])])
        if 'val_start' in kwargs:
            new_argv.extend(['--val_start', kwargs['val_start']])
        if 'val_end' in kwargs:
            new_argv.extend(['--val_end', kwargs['val_end']])
        if 'skip_training' in kwargs and kwargs['skip_training']:
            new_argv.append('--skip_training')
        if 'skip_evaluation' in kwargs and kwargs['skip_evaluation']:
            new_argv.append('--skip_evaluation')
        if 'skip_forecast' in kwargs and kwargs['skip_forecast']:
            new_argv.append('--skip_forecast')
        
        # sys.argv 완전히 대체
        sys.argv = new_argv
        
        from src.pipeline.run_all import main
        main()
    finally:
        # 원본 sys.argv 복원
        sys.argv = original_argv


if __name__ == "__main__":
    # 기본 실행
    run()
    
    # 또는 인자와 함께:
    # run(config_dir="config", val_mode="range", val_start="2023-01-01", val_end="2023-12-01")
