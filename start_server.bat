@echo off
echo Starting TestPilot Live Web Server on http://localhost:8000 ...
.\.venv\Scripts\python.exe -m uvicorn testpilot.web.api:app --host 0.0.0.0 --port 8000 --reload
pause

