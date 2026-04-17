import sys
from pathlib import Path

# Adiciona o diretório src ao sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
