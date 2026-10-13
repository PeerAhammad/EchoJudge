"""Run: uvicorn backend.main:app --host 127.0.0.1 --port 8000   (then open http://127.0.0.1:8000)"""
from backend.api.app import create_app

app = create_app()
