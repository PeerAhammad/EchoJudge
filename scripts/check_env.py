"""Report Python/library versions and whether CUDA is usable (GPU is optional)."""
import importlib, platform, sys

print("Python:", platform.python_version(), "|", platform.platform())
for m in ["numpy", "scipy", "librosa", "soundfile", "parselmouth", "pydantic", "yaml", "fastapi", "pytest"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} OK  {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  {m:12s} MISSING ({type(e).__name__}) -> pip install -r requirements.txt")
try:
    import torch
    print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except ImportError:
    print("torch not installed (only needed from Stage 2 for Whisper/alignment; CPU path works without it)")
