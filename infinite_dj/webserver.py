"""
Tiny dependency-free web server for the local player.

Serves the static player (webplayer/), the rendered audio (with HTTP Range so
the browser can seek), and the timeline JSON. No framework — just http.server.
"""

import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEBPLAYER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webplayer")

_CTYPE = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg",
}


def _make_handler(audio_path, timeline_path):
    audio_ctype = _CTYPE.get(os.path.splitext(audio_path)[1].lower(), "audio/wav")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):   # keep the console quiet
            pass

        def _send_static(self, name):
            path = os.path.join(WEBPLAYER_DIR, name)
            if not os.path.isfile(path):
                self.send_error(404); return
            ctype = _CTYPE.get(os.path.splitext(name)[1].lower(), "text/plain")
            self._send_whole(path, ctype)

        def _send_whole(self, path, ctype):
            size = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb") as f:
                try:
                    shutil.copyfileobj(f, self.wfile)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _send_ranged(self, path, ctype):
            size = os.path.getsize(path)
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                s, _, e = rng[6:].partition("-")
                start = int(s) if s else 0
                end = int(e) if e else size - 1
                end = min(end, size - 1)
                length = max(0, end - start + 1)
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    try:
                        while remaining > 0:
                            chunk = f.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
            else:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(path, "rb") as f:
                    try:
                        shutil.copyfileobj(f, self.wfile)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        def do_GET(self):
            route = self.path.split("?", 1)[0]
            if route in ("/", "/index.html"):
                self._send_static("index.html")
            elif route in ("/player.css", "/player.js"):
                self._send_static(route.lstrip("/"))
            elif route == "/timeline.json":
                self._send_whole(timeline_path, _CTYPE[".json"])
            elif route == "/audio":
                self._send_ranged(audio_path, audio_ctype)
            else:
                self.send_error(404)

    return Handler


def serve_player(audio_path, timeline_path, port=8765, host="127.0.0.1"):
    """Start the player server. Returns (server, url); call server.serve_forever()."""
    handler = _make_handler(os.path.abspath(audio_path), os.path.abspath(timeline_path))
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    return httpd, url
