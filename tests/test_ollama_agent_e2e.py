"""End-to-end HTTP-to-parser-to-agent test; uses fake Ollama, NOT Qwen."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from resource_ai.agent import repair
from resource_ai.llm import OllamaLLM
from resource_ai.runner import run_script


class RepairReply(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass
    def do_POST(self):
        _ = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        if self.path != '/api/chat':
            self.send_error(404)
            return
        patch = ('experiment.py\n<<<<<<< SEARCH\nmax_features="auto"\n'
                 '=======\nmax_features=1.0\n>>>>>>> REPLACE')
        body = json.dumps({'message': {'content': patch},
                           'prompt_eval_count': 100, 'eval_count': 25}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_http_model_response_applied_and_verified(tmp_path):
    root = Path(__file__).resolve().parents[1]
    case = root / 'benchmark' / 'cases' / 'RF02_max_features_auto_reg'
    reference = run_script(case / 'reference' / 'experiment.py', timeout=90)
    assert reference.ok and reference.trace, reference.stderr
    server = ThreadingHTTPServer(('127.0.0.1', 0), RepairReply)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        reply = repair(case / 'broken', reference.trace,
                       OllamaLLM('fake-coder:7b', host=f'http://127.0.0.1:{server.server_port}'),
                       mode='guarded+kb', max_iters=1, timeout=90,
                       workdir=tmp_path / 'fixed_copy', log=lambda *x: None)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)
    assert reply.accepted
    assert reply.final_verdict == 'REPRODUCED'
    assert reply.llm_calls == 1
    assert reply.prompt_tokens == 100 and reply.completion_tokens == 25
    assert 'max_features=1.0' in reply.diff
