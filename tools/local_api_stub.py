#!/usr/bin/env python
import json, time, random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = '127.0.0.1'
PORT = 11434

PHRASES = [
    "blue shirt, black pants, sneakers, backpack",
    "red jacket, dark jeans, boots, no bag",
    "white t-shirt, gray trousers, sneakers, shoulder bag",
    "green hoodie, black shorts, sandals, no accessories",
    "black coat, blue jeans, loafers, handbag"
]

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        raw = json.dumps(obj)
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw.encode('utf-8'))

    def log_message(self, fmt, *args):
        # quiet output
        return

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0') or '0')
        body = self.rfile.read(length).decode('utf-8') if length > 0 else '{}'
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}
        # pick a phrase
        phrase = random.choice(PHRASES)
        if self.path.endswith('/v1/chat/completions'):
            resp = {
                "id": f"stub-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get('model') or 'gpt-4o-mini',
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": phrase},
                        "finish_reason": "stop"
                    }
                ]
            }
            self._send_json(resp, 200)
        elif self.path.endswith('/v1/messages'):
            resp = {
                "output": {
                    "choices": [
                        {
                            "content": [
                                {"type": "output_text", "text": phrase}
                            ]
                        }
                    ]
                }
            }
            self._send_json(resp, 200)
        else:
            self._send_json({"error": "not_found"}, 404)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[STUB] listening on http://{HOST}:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()