@echo off
echo Starting UrbanTransit-Assistant API...
set PYTHONPATH=%PYTHONPATH%;%CD%
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
pause
