"""
Script de entrenamiento para modelos SMS (Smishing).
Usa el pipeline unificado de src/training.
"""
import sys
from pathlib import Path

# Add project root to python path to ensure imports work
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.pipeline import TrainingPipeline

def main():
    pipeline = TrainingPipeline()
    pipeline.run(data_type='sms')

if __name__ == "__main__":
    main()
