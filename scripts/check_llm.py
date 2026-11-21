"""Check a running local Ollama server with explicit model-tag and generation tests."""
import argparse
import json
from resource_ai.llm import probe_ollama


def main(argv=None):
    parser = argparse.ArgumentParser(description='Diagnose local Ollama/Qwen setup')
    parser.add_argument('--model', default='qwen2.5-coder:7b')
    parser.add_argument('--host', default=None)
    parser.add_argument('--smoke', action='store_true', help='Actually generate a small model response')
    parser.add_argument('--timeout', type=int, default=90)
    opts = parser.parse_args(argv)
    result = probe_ollama(opts.model, opts.host, smoke=opts.smoke, timeout=opts.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result['model_installed'] and (not opts.smoke or result['completion_ok']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
