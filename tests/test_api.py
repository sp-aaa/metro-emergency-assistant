import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the app (adjust import based on file structure)
import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "UrbanTransit-Assistant"}

@patch('ollama.chat')
def test_chat_endpoint(mock_chat):
    # Mock Ollama response
    mock_chat.return_value = {
        'message': {'content': '这是模拟的回答'}
    }
    
    payload = {
        "message": "测试消息",
        "history": []
    }
    
    response = client.post("/api/v1/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data['response'] == '这是模拟的回答'
    assert len(data['context']) > 0

@patch('ollama.chat')
def test_chat_error_handling(mock_chat):
    # Mock exception
    mock_chat.side_effect = Exception("Ollama connection failed")
    
    payload = {"message": "fail"}
    response = client.post("/api/v1/chat", json=payload)
    
    assert response.status_code == 500
    assert "Ollama connection failed" in response.json()['detail']
