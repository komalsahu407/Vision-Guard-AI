"""VisionGuard dashboard server (standard library only, no extra install needed).

Run:  python app.py        then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2

import config
from core import analysis, detector, notify, pipeline, webcopy

WEB = config.ROOT / "web"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
MAX_UPLOAD = 1 << 30   # 1 GB
SETTINGS_FILE = config.ROOT / "data" / "settings.json"
SETTING_LIMITS = {"conf": (0.15, 0.8), "bag_conf": (0.05, 0.6), "stride": (1, 4)}
SETTINGS = {"conf": config.CONF_THRESHOLD, "bag_conf": config.BAG_CONF_THRESHOLD, "stride": config.FRAME_STRIDE, "allow_raw": True}
DEFAULT_SETTINGS = dict(SETTINGS)


def apply_settings(new: dict) -> dict:
    for k, (lo, hi) in SETTING_LIMITS.items():
        if k in new:
            try:
                SETTINGS[k] = int(new[k]) if k == "stride" else round(max(lo, min(hi, float(new[k]))), 2)
            except (TypeError, ValueError):
                pass
    if "allow_raw" in new:
        SETTINGS["allow_raw"] = bool(new["allow_raw"])
    config.CONF_THRESHOLD, config.BAG_CONF_THRESHOLD, config.FRAME_STRIDE = SETTINGS["conf"], SETTINGS["bag_conf"], SETTINGS["stride"]
    return dict(SETTINGS)


def load_settings():
    try:
        apply_settings(json.loads(SETTINGS_FILE.read_text()))
    except Exception:
        pass


def dir_mb(p: Path) -> float:
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1) if p.exists() else 0.0


# ------------------------------------------------------------------ helpers
def system_info() -> dict:
    """Real CPU / RAM numbers of this laptop. Uses psutil when installed, else best-effort fallbacks."""
    info = {"cores": os.cpu_count(), "cpu": None, "ram_pct": None, "ram_used_gb": None, "ram_total_gb": None,
            "proc_mb": None, "source": "none"}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info.update(cpu=psutil.cpu_percent(interval=None), ram_pct=vm.percent,
                    ram_used_gb=round((vm.total - vm.available) / 1e9, 2), ram_total_gb=round(vm.total / 1e9, 2),
                    proc_mb=round(psutil.Process().memory_info().rss / 1e6), source="psutil")
        return info
    except ImportError:
        pass
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                            ("ae", ctypes.c_ulonglong)]
            m = MS(); m.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            info.update(ram_pct=float(m.load), ram_total_gb=round(m.tp / 1e9, 2), ram_used_gb=round((m.tp - m.ap) / 1e9, 2), source="ctypes")
        else:
            mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
            tot, av = mem["MemTotal"] * 1024, mem["MemAvailable"] * 1024
            info.update(ram_pct=round(100 * (tot - av) / tot, 1), ram_total_gb=round(tot / 1e9, 2), ram_used_gb=round((tot - av) / 1e9, 2), source="proc")
    except Exception:
        pass
    return info


def list_videos() -> list[dict]:
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(config.VIDEO_DIR.iterdir()):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            continue
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not w or not h:
            continue
        cdir = pipeline.cache_dir_for(p)
        out.append({
            "name": p.name, "stem": p.stem, "fps": round(fps, 2), "duration": round(n / fps, 2) if n else 0,
            "proc_size": [config.PROC_WIDTH, int(round(h * config.PROC_WIDTH / w / 2) * 2)],
            "detected": (cdir / "meta.json").exists(), "analyzed": (cdir / "analysis.json").exists(),
            "web_ready": (not webcopy.needs_copy(p)) or webcopy.web_path(cdir).exists(),
            "cache_mb": dir_mb(cdir),
            "has_gt": (config.VIDEO_DIR / f"{p.stem}.gt.json").exists(),
        })
    return out


def video_path(name: str) -> Path | None:
    p = config.VIDEO_DIR / Path(name).name
    return p if p.exists() and p.suffix.lower() in VIDEO_EXT else None


def set_job(stem: str, **kw):
    with JOBS_LOCK:
        JOBS.setdefault(stem, {"state": "idle", "progress": 0.0, "message": "", "error": None}).update(kw)


def run_job(name: str, force: bool, params: dict):
    p = video_path(name)
    stem = p.stem
    try:
        set_job(stem, state="running", progress=0.0, message="Starting", error=None)
        cdir = pipeline.cache_dir_for(p)
        set_job(stem, message="Preparing browser-playable copy")
        wc = webcopy.ensure(p, cdir)
        if wc.startswith("failed"):
            set_job(stem, message="Video preview unavailable (" + wc[7:] + "); analysis continues")
        need_detect = force or not (cdir / "tracks.json").exists()
        if not need_detect:
            # old caches (Step 1) have no boxes; redo detection automatically
            data = json.loads((cdir / "tracks.json").read_text())
            first = data["tracks"]
            need_detect = (bool(first) and "boxes" not in first[0]) or data["meta"].get("sig") != detector.signature()
        if need_detect:
            set_job(stem, message="Detecting and tracking (YOLO)")
            pipeline.run(p, force=True, on_progress=lambda pr, m: set_job(stem, progress=round(pr * 0.85, 3), message=f"Detecting: {m}"))
        set_job(stem, progress=0.9, message="Scoring incidents and building evidence")
        analysis.analyze(p.name, params)
        msg = "Done"
        try:
            if notify.load()["auto"]:
                r = notify.send_for_video(p.name, cdir)
                msg = f"Done. Alerts sent: {r['sent']}, entry notices: {r['entries']}" + (f" ({len(r['errors'])} problem(s))" if r["errors"] else "")
        except Exception as ex:
            msg = f"Done (alert sending failed: {ex})"
        set_job(stem, state="done", progress=1.0, message=msg)
    except Exception as e:  # shown in the UI
        traceback.print_exc()
        set_job(stem, state="error", error=f"{type(e).__name__}: {e}", message="Failed")


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "VisionGuard/1.0"

    def log_message(self, *a):  # keep terminal quiet
        pass

    # ---- responses
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def _file(self, path: Path, ctype=None, ranged=False):
        size = path.stat().st_size
        ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if ranged and rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if not ranged:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ---- routing
    def do_GET(self):
        u = urlparse(self.path)
        parts = [unquote(x) for x in u.path.strip("/").split("/")]
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._file(WEB / "index.html", "text/html; charset=utf-8")
            if parts[0] != "api":
                return self._err(404, "not found")
            route = parts[1] if len(parts) > 1 else ""
            arg = parts[2] if len(parts) > 2 else ""
            if route == "videos":
                return self._json({"videos": list_videos(), "detector_ready": config.MODEL_PATH.exists()})
            if route == "settings":
                return self._json({"settings": dict(SETTINGS), "defaults": DEFAULT_SETTINGS, "model": detector.resolve_model_path().name,
                                   "model_ready": detector.resolve_model_path().exists()})
            if route == "notify":
                d = notify.public(notify.load()); d["last"] = dict(notify.LAST)
                return self._json(d)
            if route == "ack":
                p = video_path(arg)
                f = pipeline.cache_dir_for(p) / "ack.json" if p else None
                return self._json(json.loads(f.read_text()) if f and f.exists() else {})
            if route == "system":
                d = system_info()
                with JOBS_LOCK:
                    d["jobs_running"] = sum(1 for j in JOBS.values() if j.get("state") == "running")
                return self._json(d)
            if route == "video":
                p = video_path(arg)
                if not p:
                    return self._err(404, "video not found")
                w = webcopy.web_path(pipeline.cache_dir_for(p))
                return self._file(w if w.exists() else p, ranged=True)
            if route == "frame":
                p = video_path(arg)
                if not p:
                    return self._err(404, "video not found")
                cap = cv2.VideoCapture(str(p))
                ok, fr = cap.read()
                cap.release()
                if not ok:
                    return self._err(500, "cannot read frame")
                h, w = fr.shape[:2]
                fr = cv2.resize(fr, (config.PROC_WIDTH, int(round(h * config.PROC_WIDTH / w / 2) * 2)))
                data = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if route in ("tracks", "result", "zone", "status"):
                p = video_path(arg)
                if not p:
                    return self._err(404, "video not found")
                cdir = pipeline.cache_dir_for(p)
                if route == "status":
                    with JOBS_LOCK:
                        return self._json(dict(JOBS.get(p.stem, {"state": "idle", "progress": 0, "message": "", "error": None})))
                f = {"tracks": "tracks.json", "result": "analysis.json", "zone": "zone.json"}[route]
                if not (cdir / f).exists():
                    return self._json({"empty": True}) if route == "zone" else self._err(404, "not available yet")
                return self._file(cdir / f, "application/json")
            if route == "evidence":
                p = video_path(arg)
                fn = parts[3] if len(parts) > 3 else ""
                if not p or not re.fullmatch(r"inc\d+(_raw)?\.jpg", fn):
                    return self._err(404, "not found")
                raw = q.get("raw", ["0"])[0] == "1" and SETTINGS["allow_raw"]
                if raw and not fn.endswith("_raw.jpg"):
                    fn = fn[:-4] + "_raw.jpg"
                f = pipeline.cache_dir_for(p) / "evidence" / fn
                return self._file(f, "image/jpeg") if f.exists() else self._err(404, "no evidence image")
            return self._err(404, "unknown route")
        except Exception as e:
            traceback.print_exc()
            return self._err(500, f"{type(e).__name__}: {e}")

    def do_POST(self):
        u = urlparse(self.path)
        parts = [unquote(x) for x in u.path.strip("/").split("/")]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if parts[:2] == ["api", "upload"]:
                name = Path(unquote(self.headers.get("X-Filename", ""))).name
                if Path(name).suffix.lower() not in VIDEO_EXT:
                    return self._err(400, "Please choose a video file (.mp4, .mov, .avi, .mkv, .webm).")
                if n <= 0 or n > MAX_UPLOAD:
                    return self._err(400, "File is empty or larger than 1 GB.")
                safe = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
                config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
                tmp = config.VIDEO_DIR / (safe + ".part")
                left = n
                with open(tmp, "wb") as f:
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
                if left:
                    tmp.unlink(missing_ok=True)
                    return self._err(400, "Upload was interrupted. Try again.")
                tmp.replace(config.VIDEO_DIR / safe)
                return self._json({"name": safe})
            body = json.loads(self.rfile.read(n) or b"{}")
            if parts[:2] == ["api", "settings"]:
                s = apply_settings(body)
                SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
                SETTINGS_FILE.write_text(json.dumps(s))
                return self._json({"settings": s})
            if parts[:2] == ["api", "notify"]:
                cfg_ = notify.load()
                if len(parts) > 2 and parts[2] == "test":
                    ch = body.get("channel")
                    try:
                        if ch == "email":
                            notify.send_email(cfg_, "[VisionGuard] Test message", "This is a test from VisionGuard. Email alerts are working.")
                        elif ch == "whatsapp":
                            notify.send_whatsapp(cfg_, "VisionGuard test: WhatsApp alerts are working.")
                        else:
                            return self._err(400, "unknown channel")
                        return self._json({"ok": True})
                    except Exception as ex:
                        return self._err(400, f"{type(ex).__name__}: {str(ex)[:200]}")
                if len(parts) > 2 and parts[2] == "send":
                    p = video_path(body.get("video", ""))
                    return self._json(notify.send_for_video(p.name, pipeline.cache_dir_for(p))) if p else self._err(404, "video not found")
                notify.update(body)
                return self._json(notify.public(notify.load()))
            if parts[:2] == ["api", "ack"]:
                p = video_path(body.get("video", ""))
                if not p:
                    return self._err(404, "video not found")
                cdir = pipeline.cache_dir_for(p); cdir.mkdir(parents=True, exist_ok=True)
                f = cdir / "ack.json"
                d = json.loads(f.read_text()) if f.exists() else {}
                key, state = str(body.get("key", ""))[:120], body.get("state")
                if state in ("ack", "dismissed"):
                    d[key] = state
                else:
                    d.pop(key, None)
                f.write_text(json.dumps(d))
                return self._json(d)
            if parts[:3] == ["api", "cache", "clear"]:
                p = video_path(body.get("video", ""))
                if not p:
                    return self._err(404, "video not found")
                with JOBS_LOCK:
                    if JOBS.get(p.stem, {}).get("state") == "running":
                        return self._err(409, "analysis is running for this video")
                cdir = pipeline.cache_dir_for(p)
                import shutil
                for name in ("tracks.json", "meta.json", "analysis.json", "preview.mp4", "ack.json", "notified.json"):
                    (cdir / name).unlink(missing_ok=True)
                shutil.rmtree(cdir / "evidence", ignore_errors=True)
                return self._json({"cleared": True})
            if parts[:2] == ["api", "analyze"]:
                p = video_path(body.get("video", ""))
                if not p:
                    return self._err(404, "video not found")
                with JOBS_LOCK:
                    if JOBS.get(p.stem, {}).get("state") == "running":
                        return self._err(409, "already running")
                set_job(p.stem, state="running", progress=0.0, message="Queued", error=None)
                threading.Thread(target=run_job, args=(p.name, bool(body.get("force")), body.get("params") or {}),
                                 daemon=True).start()
                return self._json({"started": True})
            if parts[:2] == ["api", "zone"] and len(parts) > 2:
                p = video_path(parts[2])
                if not p:
                    return self._err(404, "video not found")
                pts = body.get("points") or []
                if pts and len(pts) < 3:
                    return self._err(400, "a zone needs at least 3 points")
                cdir = pipeline.cache_dir_for(p)
                cdir.mkdir(parents=True, exist_ok=True)
                (cdir / "zone.json").write_text(json.dumps({"points": pts, "name": str(body.get("name") or "Restricted zone")[:40]}))
                return self._json({"saved": True})
            return self._err(404, "unknown route")
        except json.JSONDecodeError:
            return self._err(400, "bad JSON")
        except Exception as e:
            traceback.print_exc()
            return self._err(500, f"{type(e).__name__}: {e}")


def main(port: int = 8000):
    load_settings()
    try:
        import psutil
        psutil.cpu_percent(interval=None)   # prime the counter
    except ImportError:
        pass
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"VisionGuard running at {url}   (press Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
