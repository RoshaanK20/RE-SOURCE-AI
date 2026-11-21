"""No network needed: verify exactly how RE:SOURCE talks to Ollama."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from resource_ai.llm import (LLMModelError, OllamaLLM, probe_ollama)


class FakeOllama(BaseHTTPRequestHandler):
    model_names = ['qwen2.5-coder:7b']
    fail_chat = False
    def log_message(self, *_):
        pass
    def _reply(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == '/api/tags':
            self._reply({'models':[{'name': name} for name in self.model_names]})
        else:
            self._reply({'error':'not found'}, 404)
    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        body = json.loads(self.rfile.read(length))
        if self.path != '/api/chat':
            self._reply({'error': 'not found'}, 404)
        elif body['model'] not in self.model_names:
            self._reply({'error': 'model not found'}, 404)
        elif self.fail_chat:
            self._reply({'message': {'content': ''}})
        else:
            self._reply({'message': {'content': 'READY'},
                         'prompt_eval_count': 17, 'eval_count': 1})


@pytest.fixture
def server():
    FakeOllama.fail_chat = False
    http = ThreadingHTTPServer(('127.0.0.1', 0), FakeOllama)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}'
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=3)


def test_tags_and_live_completion_protocol(server):
    out = probe_ollama('qwen2.5-coder:7b', server, smoke=True)
    assert out['server_reachable']
    assert out['model_installed']
    assert out['completion_tested'] and out['completion_ok']
    assert out['completion_excerpt'] == 'READY'


def test_missing_model_explains_pull_command(server):
    p = probe_ollama('qwen2.5-coder:3b', server, smoke=True)
    assert p['stage'] == 'model_missing'
    assert not p['completion_tested']
    assert 'ollama pull qwen2.5-coder:3b' in p['error']
    with pytest.raises(LLMModelError, match='ollama pull'):
        OllamaLLM('missing:1b', host=server).chat([{'role':'user','content':'hello'}])


def test_connection_failure_reports_usable_error():
    # Port 1 is reserved, not a running Ollama server on ordinary systems.
    p = probe_ollama('qwen2.5-coder:7b', 'http://127.0.0.1:1')
    assert p['stage'] == 'connection' and not p['server_reachable']


def test_empty_response_is_not_success(server):
    from resource_ai.llm import LLMResponseError
    FakeOllama.fail_chat = True
    with pytest.raises(LLMResponseError, match='empty completion'):
        OllamaLLM('qwen2.5-coder:7b', host=server).chat([{'role':'user','content':'hello'}])
