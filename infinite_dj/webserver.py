"""
Tiny dependency-free web app for the local studio + player.

Serves the setup pane, the player, the library listing, and renders mixes on
demand in a background thread. Rendered artifacts (audio + timeline JSON) are
written into an output directory and served back per job id, with HTTP Range
support so the browser can seek. No framework — just http.server.
"""

import json
import os
import shutil
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEBPLAYER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webplayer")

_CTYPE = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg",
}
_STATIC = {"/player.css", "/player.js", "/setup.css", "/setup.js"}

# job_id -> {status: queued|rendering|done|error, audio, timeline, error, spec}
_jobs: dict = {}
_jobs_lock = threading.Lock()

# job_id -> RadioSession (endless mixes; separate lifecycle from one-shot jobs)
_radios: dict = {}
_radios_lock = threading.Lock()


def _radio(job_id):
    with _radios_lock:
        return _radios.get(job_id)


def _job(job_id):
    with _jobs_lock:
        return dict(_jobs.get(job_id) or {})


def _set_job(job_id, **fields):
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _render_job(job_id, spec, db_path, output_dir):
    """Background worker: render the spec and write audio + timeline."""
    try:
        import soundfile as sf
        from .db import TrackDB
        from .mixspec import render_from_spec
        from .timeline import write_timeline

        _set_job(job_id, status="rendering")
        db = TrackDB(db_path)          # own connection: sqlite is per-thread
        tracks = db.load_all()
        db.close()

        audio, sr, clips, pool = render_from_spec(spec, tracks)

        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, f"{job_id}.wav")
        tl_path = os.path.join(output_dir, f"{job_id}.timeline.json")
        sf.write(audio_path, audio, sr, subtype="PCM_16")
        write_timeline(tl_path, clips, pool, len(audio) / sr, sr)

        _set_job(job_id, status="done", audio=audio_path, timeline=tl_path,
                 duration=len(audio) / sr)
    except Exception as e:                                  # surface to the UI
        _set_job(job_id, status="error", error=str(e))


def _make_handler(db_path, output_dir):

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):   # keep the console quiet
            pass

        # ── helpers ──────────────────────────────────────────────────────────
        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", _CTYPE[".json"])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_static(self, name):
            path = os.path.join(WEBPLAYER_DIR, name)
            if not os.path.isfile(path):
                self.send_error(404); return
            self._send_whole(path, _CTYPE.get(os.path.splitext(name)[1].lower(),
                                              "text/plain"))

        def _send_whole(self, path, ctype):
            if not os.path.isfile(path):
                self.send_error(404); return
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
            if not os.path.isfile(path):
                self.send_error(404); return
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

        def _query(self):
            from urllib.parse import parse_qs, urlparse
            return parse_qs(urlparse(self.path).query)

        def _job_artifact(self, key):
            """Resolve ?job=<id> to a stored artifact path ('audio'|'timeline')."""
            q = self._query()
            job_id = (q.get("job") or [None])[0] or "default"
            return _job(job_id).get(key)

        # ── routes ───────────────────────────────────────────────────────────
        def do_GET(self):
            route = self.path.split("?", 1)[0]

            if route in ("/", "/index.html", "/setup"):
                # Without a DB there's nothing to browse — go straight to the player.
                self._send_static("setup.html" if db_path else "player.html")
            elif route == "/player":
                self._send_static("player.html")
            elif route in _STATIC:
                self._send_static(route.lstrip("/"))
            elif route == "/api/library":
                if not db_path:
                    self._send_json({"artists": []}); return
                try:
                    from .db import TrackDB
                    from .mixspec import PACE_PRESETS, library_groups
                    db = TrackDB(db_path); tracks = db.load_all(); db.close()
                    self._send_json({
                        "artists": library_groups(tracks),
                        "pace_presets": sorted(PACE_PRESETS),
                        "count": len(tracks),
                    })
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            elif route == "/api/render":
                q = self._query()
                job_id = (q.get("job") or [None])[0]
                if not job_id:
                    self._send_json({"error": "missing job"}, 400); return
                j = _job(job_id)
                if not j:
                    self._send_json({"error": "unknown job"}, 404); return
                self._send_json({"job": job_id, "status": j.get("status"),
                                 "error": j.get("error"),
                                 "duration": j.get("duration")})
            elif route == "/api/radio/state":
                q = self._query()
                s = _radio((q.get("job") or [None])[0])
                if not s:
                    self._send_json({"error": "unknown radio session"}, 404); return
                played = (q.get("played") or [None])[0]
                if played is not None:
                    try:
                        s.update_playback_position(float(played))
                    except (TypeError, ValueError):
                        self._send_json({"error": "invalid playback position"}, 400); return
                self._send_json(s.state())
            elif route == "/api/radio/chunk":
                q = self._query()
                s = _radio((q.get("job") or [None])[0])
                try:
                    idx = int((q.get("i") or ["-1"])[0])
                except ValueError:
                    idx = -1
                path = s.chunk_path(idx) if s else None
                if not path:
                    self.send_error(404); return
                self._send_ranged(path, _CTYPE[".wav"])
            elif route == "/timeline.json":
                path = self._job_artifact("timeline")
                self._send_whole(path, _CTYPE[".json"]) if path else self.send_error(404)
            elif route == "/audio":
                path = self._job_artifact("audio")
                if not path:
                    self.send_error(404); return
                ctype = _CTYPE.get(os.path.splitext(path)[1].lower(), "audio/wav")
                self._send_ranged(path, ctype)
            else:
                self.send_error(404)

        def _read_spec(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_POST(self):
            route = self.path.split("?", 1)[0]

            if route == "/api/radio/stop":
                q = self._query()
                s = _radio((q.get("job") or [None])[0])
                if s:
                    s.stop()
                    s.cleanup()
                    with _radios_lock:
                        _radios.pop(s.job_id, None)
                self._send_json({"stopped": True}); return

            if route == "/api/radio":
                if not db_path:
                    self._send_json({"error": "server has no library"}, 400); return
                try:
                    spec = self._read_spec()
                except Exception as e:
                    self._send_json({"error": f"bad request: {e}"}, 400); return
                try:
                    from .db import TrackDB
                    from .mixspec import (radio_profile, radio_supported,
                                          select_tracks)
                    from .radio import RadioSession
                    if not radio_supported(spec):
                        self._send_json(
                            {"error": "radio isn't available at INSANE yet"}, 400)
                        return
                    db = TrackDB(db_path); tracks = db.load_all(); db.close()
                    pool = select_tracks(spec, tracks)
                    job_id = uuid.uuid4().hex[:12]
                    session = RadioSession(pool, output_dir, job_id,
                                           params=radio_profile(spec))
                    with _radios_lock:
                        _radios[job_id] = session
                    # Prime synchronously so the client can start immediately,
                    # then let the lookahead run ahead of the listener.
                    session.prime()
                    session.start()
                    self._send_json({"job": job_id, **session.state()})
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
                return

            if route != "/api/render":
                self.send_error(404); return
            if not db_path:
                self._send_json({"error": "server has no library"}, 400); return
            try:
                spec = self._read_spec()
            except Exception as e:
                self._send_json({"error": f"bad request: {e}"}, 400); return

            job_id = uuid.uuid4().hex[:12]
            _set_job(job_id, status="queued", spec=spec)
            threading.Thread(target=_render_job,
                             args=(job_id, spec, db_path, output_dir),
                             daemon=True).start()
            self._send_json({"job": job_id, "status": "queued"})

    return Handler


def serve_app(db_path, output_dir, port=8765, host="127.0.0.1"):
    """Start the studio app (setup pane + render-on-demand + player)."""
    handler = _make_handler(os.path.abspath(db_path) if db_path else None,
                            os.path.abspath(output_dir))
    httpd = ThreadingHTTPServer((host, port), handler)
    return httpd, f"http://{host}:{port}/"


def serve_player(audio_path, timeline_path, port=8765, host="127.0.0.1"):
    """
    Serve a single pre-rendered mix (the `dj.py serve` path). The artifacts are
    registered as the "default" job so the player's ?job-less requests find them.
    """
    _set_job("default", status="done",
             audio=os.path.abspath(audio_path),
             timeline=os.path.abspath(timeline_path))
    handler = _make_handler(None, os.path.dirname(os.path.abspath(audio_path)))
    httpd = ThreadingHTTPServer((host, port), handler)
    return httpd, f"http://{host}:{port}/"
