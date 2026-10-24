"""Send incident alerts by email (SMTP, e.g. Gmail app password) and WhatsApp (CallMeBot, text only).

Secrets live ONLY in data/notify.json on this laptop. They are never sent to the browser and never zipped.
Only the privacy-blurred evidence frame is ever attached.
"""
from __future__ import annotations

import json
import smtplib
import ssl
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path

import cv2
import numpy as np

import config

FILE = config.ROOT / "data" / "notify.json"
LEVELS = {"low": 0, "medium": 1, "high": 2}
ENTRY_MIN_S = 1.0     # seconds inside the zone before an entry notice is sent (brushing the edge is ignored)
LAST: dict = {}       # result of the latest send, shown in the dashboard
DEFAULTS = {
    "min_level": "low", "auto": True, "entries": True,
    "email": {"enabled": False, "host": "smtp.gmail.com", "port": 465, "security": "ssl", "user": "", "password": "", "to": ""},
    "whatsapp": {"enabled": False, "phone": "", "apikey": "", "base_url": "https://api.callmebot.com/whatsapp.php"},
}


def load() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    try:
        d = json.loads(FILE.read_text())
        for k in ("min_level", "auto", "entries"):
            if k in d:
                cfg[k] = d[k]
        for ch in ("email", "whatsapp"):
            cfg[ch].update(d.get(ch, {}))
    except Exception:
        pass
    return cfg


def save(cfg: dict) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(cfg, indent=2))


def public(cfg: dict) -> dict:
    """What the browser may see: no password, no API key."""
    out = json.loads(json.dumps(cfg))
    out["email"]["has_password"] = bool(out["email"].pop("password", ""))
    out["whatsapp"]["has_key"] = bool(out["whatsapp"].pop("apikey", ""))
    return out


def update(new: dict) -> dict:
    cfg = load()
    if new.get("min_level") in LEVELS:
        cfg["min_level"] = new["min_level"]
    for k in ("auto", "entries"):
        if k in new:
            cfg[k] = bool(new[k])
    for ch, secret in (("email", "password"), ("whatsapp", "apikey")):
        src = new.get(ch) or {}
        for k in cfg[ch]:
            if k not in src:
                continue
            v = src[k]
            if k == "enabled":
                cfg[ch][k] = bool(v)
            elif k == "port":
                try:
                    cfg[ch][k] = int(v)
                except (TypeError, ValueError):
                    pass
            elif k == secret:
                if str(v).strip():          # empty = keep the saved secret
                    cfg[ch][k] = str(v).strip().replace(" ", "")
            elif k == "base_url":
                continue                    # not editable from the browser
            else:
                cfg[ch][k] = str(v).strip()
    save(cfg)
    return cfg


def send_email(cfg: dict, subject: str, body: str, image: Path | None = None) -> None:
    e = cfg["email"]
    if not (e["user"] and e["password"] and e["to"]):
        raise ValueError("Email is not set up: fill your Gmail address, app password and the address to send to.")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = e["user"], e["to"], subject
    msg.set_content(body)
    if image and image.exists():
        msg.add_attachment(image.read_bytes(), maintype="image", subtype="jpeg", filename="evidence_blurred.jpg")
    if e["security"] == "ssl":
        with smtplib.SMTP_SSL(e["host"], e["port"], timeout=25, context=ssl.create_default_context()) as s:
            s.login(e["user"], e["password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(e["host"], e["port"], timeout=25) as s:
            if e["security"] == "starttls":
                s.starttls(context=ssl.create_default_context())
            if e["security"] != "none":
                s.login(e["user"], e["password"])
            s.send_message(msg)


def send_whatsapp(cfg: dict, text: str) -> None:
    w = cfg["whatsapp"]
    if not (w["phone"] and w["apikey"]):
        raise ValueError("WhatsApp is not set up: fill your number (with country code) and the CallMeBot API key.")
    q = urllib.parse.urlencode({"phone": w["phone"].replace("+", "").replace(" ", ""), "text": text[:900], "apikey": w["apikey"]})
    with urllib.request.urlopen(f"{w['base_url']}?{q}", timeout=25) as r:
        body = r.read().decode("utf-8", "replace")
        if r.status >= 300 or "error" in body.lower()[:200]:
            raise RuntimeError("WhatsApp service said: " + body[:160].replace("\n", " "))


def describe(video: str, inc: dict) -> tuple[str, str]:
    t0, t1 = inc["start_t"], inc["end_t"]
    subj = f"[VisionGuard] {inc['level'].upper()} {round(inc['severity'])}/100: {inc['type']}"
    body = (f"VisionGuard alert\nVideo: {video}\nWhat: {inc['type']}\nSeverity: {round(inc['severity'])}/100 ({inc['level']})\n"
            f"When: {t0:.1f}s to {t1:.1f}s in the video\nWhy: {inc['summary']}\n\n"
            "This is an automatic pattern flag, not a verdict. Please review the footage.")
    return subj, body


def incident_key(inc: dict) -> str:
    return f"{inc['type']}|{inc['track_id']}|{round(inc['start_t'])}"


def zone_entries(cache_dir: Path) -> list[dict]:
    """People who really stayed (>= ENTRY_MIN_S) inside the zone but did NOT become an incident. From real tracks."""
    zf, tf, af = cache_dir / "zone.json", cache_dir / "tracks.json", cache_dir / "analysis.json"
    if not (zf.exists() and tf.exists() and af.exists()):
        return []
    pts = json.loads(zf.read_text()).get("points") or []
    if len(pts) < 3:
        return []
    poly = np.array(pts, np.float32)
    data, an = json.loads(tf.read_text()), json.loads(af.read_text())
    fps, dt = data["meta"]["src_fps"], data["meta"]["stride"] / data["meta"]["src_fps"]
    kept, idm = set(an.get("kept_ids", [])), an.get("identity", {})
    pid_of = lambda t: idm.get(str(t), t)
    flagged = {pid_of(t) for inc in an.get("incidents", []) for t in (inc.get("track_ids") or [inc["track_id"]])}
    inside: dict = {}
    for t in data["tracks"]:
        if t["label"] not in config.PERSON_LABELS or t["id"] not in kept:
            continue
        pid = pid_of(t["id"])
        for f, p in zip(t["frames"], t["path"]):
            if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0:
                v = inside.setdefault(pid, {"pid": pid, "first": f / fps, "secs": 0.0})
                v["first"] = min(v["first"], f / fps)
                v["secs"] += dt
    return sorted((v for pid, v in inside.items() if v["secs"] >= ENTRY_MIN_S and pid not in flagged), key=lambda v: v["first"])


def _deliver(cfg: dict, subj: str, body: str, img: Path | None, res: dict) -> bool:
    ok = False
    for ch, fn in (("email", lambda: send_email(cfg, subj, body, img)), ("whatsapp", lambda: send_whatsapp(cfg, subj + "\n" + body))):
        if not cfg[ch]["enabled"]:
            continue
        try:
            fn()
            ok = True
        except Exception as ex:
            res["errors"].append(f"{ch}: {type(ex).__name__}: {str(ex)[:140]}")
    return ok


def send_for_video(video_name: str, cache_dir: Path) -> dict:
    """Send alerts for the CURRENT analysis run. Inside one run nothing is sent twice;
    a new Analyze is a new run, so its alerts are sent again."""
    import time
    cfg = load()
    res = {"sent": 0, "entries": 0, "already": 0, "below": 0, "errors": [], "video": video_name, "time": time.time()}
    LAST.clear(); LAST.update(res)
    af = cache_dir / "analysis.json"
    try:
        if not af.exists():
            res["errors"].append("This video has no analysis yet.")
        elif not (cfg["email"]["enabled"] or cfg["whatsapp"]["enabled"]):
            res["errors"].append("No channel is switched on (tick Email alerts or WhatsApp alerts, then Save).")
        else:
            run = af.stat().st_mtime_ns
            nf = cache_dir / "notified.json"
            st = json.loads(nf.read_text()) if nf.exists() else {}
            done = st.get("keys", {}) if st.get("run") == run else {}
            for inc in json.loads(af.read_text()).get("incidents", []):
                key = incident_key(inc)
                if LEVELS[inc["level"]] < LEVELS[cfg["min_level"]]:
                    res["below"] += 1
                elif key in done:
                    res["already"] += 1
                else:
                    subj, body = describe(video_name, inc)
                    if _deliver(cfg, subj, body, cache_dir / "evidence" / f"inc{inc['id']}.jpg", res):   # blurred frame only
                        done[key] = True
                        res["sent"] += 1
            if cfg.get("entries"):
                for e in zone_entries(cache_dir):
                    key = f"entry|{e['pid']}"
                    if key in done:
                        res["already"] += 1
                        continue
                    subj = f"[VisionGuard] Person #{e['pid']} entered the restricted zone"
                    body = (f"VisionGuard notice\nVideo: {video_name}\nPerson #{e['pid']} entered the restricted zone at {e['first']:.1f}s "
                            f"and stayed about {e['secs']:.1f}s.\nThis is an entry notice, not an incident (they left before the loiter limit).")
                    if _deliver(cfg, subj, body, None, res):
                        done[key] = True
                        res["entries"] += 1
            nf.write_text(json.dumps({"run": run, "keys": done}))
    finally:
        LAST.clear(); LAST.update(res)
    return res
