"""
FastAPI Backend — Main application
WebSocket video streaming + REST API endpoints
"""

import asyncio
import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .database import (
    init_db, add_violation, get_violations,
    get_stats, clear_violations, export_csv, EVIDENCE_DIR,
)
from .pipeline import DetectionPipeline

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR   = BASE_DIR / "uploads"

# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────
app = FastAPI(title="Traffic Violation Detection System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline instance
pipeline = DetectionPipeline(max_workers=3)


# ──────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# ENSURE DIRS EXIST BEFORE ANYTHING (incl. StaticFiles mount)
# ──────────────────────────────────────────────
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup():
    await init_db()
    print(f"✓ DB ready  | Upload: {UPLOAD_DIR} | Evidence: {EVIDENCE_DIR}")
    print(f"✓ Frontend  : {FRONTEND_DIR}")


# ──────────────────────────────────────────────
# FRONTEND — serve index.html
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    p = FRONTEND_DIR / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Frontend not found — check FRONTEND_DIR</h2>", status_code=404)


# ──────────────────────────────────────────────
# REST API — health check
# ──────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────
# REST API — Upload video
# ──────────────────────────────────────────────

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        return JSONResponse(status_code=400,
            content={"error": f"Định dạng không hỗ trợ: {ext}"})

    safe = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    path = UPLOAD_DIR / safe
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "filename": safe,
        "original_name": file.filename,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }


# ──────────────────────────────────────────────
# REST API — Violations
# ──────────────────────────────────────────────

@app.get("/api/violations")
async def api_get_violations(limit: int = 200, offset: int = 0):
    rows = await get_violations(limit, offset)
    return {"violations": rows, "count": len(rows)}


@app.get("/api/stats")
async def api_get_stats():
    return await get_stats()


@app.delete("/api/violations")
async def api_clear():
    await clear_violations()
    return {"status": "cleared"}


@app.get("/api/violations/export")
async def api_export():
    csv = await export_csv()
    return PlainTextResponse(
        content=csv, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=violations.csv"},
    )


@app.get("/api/models")
async def api_models():
    from .models_loader import models_manager
    return models_manager.all_configs()


# ──────────────────────────────────────────────
# WEBSOCKET — Realtime video stream
# ──────────────────────────────────────────────

@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected")
    
    video_path_to_clean = None
    
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            action = data.get("action", "")

            if action == "start":
                source   = data.get("source", "webcam")
                filename = data.get("filename", "")
                url      = data.get("url", "")

                video_path = ""
                if source == "video":
                    video_path = str(UPLOAD_DIR / filename)
                    if not Path(video_path).exists():
                        await ws.send_text(json.dumps(
                            {"error": f"File không tồn tại: {filename}"}))
                        continue
                elif source == "url":
                    video_path = url

                try:
                    pipeline.start(source=source, video_path=video_path)
                except RuntimeError as e:
                    await ws.send_text(json.dumps({"error": str(e)}))
                    continue

                await ws.send_text(json.dumps({
                    "status": "started",
                    "session_id": pipeline.session_id,
                    "width":  pipeline.frame_width,
                    "height": pipeline.frame_height,
                    "fps":    pipeline.source_fps,
                    "total_frames": pipeline.total_frames,
                }))
                
                if source == "video":
                    video_path_to_clean = video_path

                await _stream_loop(ws)

                # Dừng pipeline để giải phóng file handle (đặc biệt quan trọng trên ổ đĩa NAS)
                pipeline.stop()

                # Dọn rác khi video chạy xong tự nhiên
                if video_path_to_clean:
                    try:
                        p = Path(video_path_to_clean)
                        if p.exists():
                            p.unlink()
                            print(f"[Auto-Clean] Đã xóa video: {p.name}")
                        # Dọn luôn các file .nfs còn sót lại nếu có
                        for nfs_file in p.parent.glob(".nfs*"):
                            try: nfs_file.unlink()
                            except: pass
                    except Exception as e:
                        print(f"[Auto-Clean] Lỗi khi xóa: {e}")
                    video_path_to_clean = None

            elif action == "stop":
                pipeline.stop()
                await ws.send_text(json.dumps({"status": "stopped"}))

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        pipeline.stop()
        # Dọn rác nếu mất kết nối hoặc refresh đột ngột
        if video_path_to_clean:
            try:
                p = Path(video_path_to_clean)
                if p.exists():
                    p.unlink()
                    print(f"[Auto-Clean] Đã xóa video (xử lý dang dở tắt ngang): {p.name}")
            except:
                pass
            video_path_to_clean = None


async def _stream_loop(ws: WebSocket):
    loop = asyncio.get_event_loop()
    while pipeline.is_running:
        t0 = loop.time()

        result = await loop.run_in_executor(None, pipeline.process_frame)

        if result is None:
            await ws.send_text(json.dumps({"status": "finished"}))
            break

        # Binary: JPEG frame
        await ws.send_bytes(pipeline.encode_frame_jpeg(result.annotated_frame, quality=70))

        # Save & send violations
        for v in result.violations:
            await add_violation(
                track_id=v.track_id, vehicle_type=v.vehicle_type,
                violation_type=v.violation_type, confidence=v.confidence,
                frame_number=v.frame_number, bbox=v.bbox,
                evidence_path=v.evidence_path, session_id=pipeline.session_id,
            )

        progress = round(result.frame_number / pipeline.total_frames * 100, 1) \
            if pipeline.total_frames > 0 else 0

        await ws.send_text(json.dumps({
            "type": "frame_data",
            "frame": result.frame_number,
            "fps": result.fps,
            "light_status": result.light_status,
            "vehicle_count": result.vehicle_count,
            "violations": [v.to_dict() for v in result.violations],
            "total_violations_redlight": len(pipeline.red_light_checker.violators),
            "total_violations_helmet":   len(pipeline.no_helmet_checker.violated_ids),
            "progress": progress,
        }))

        # Non-blocking check for stop command
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
            if json.loads(raw).get("action") == "stop":
                pipeline.stop()
                await ws.send_text(json.dumps({"status": "stopped"}))
                break
        except (asyncio.TimeoutError, Exception):
            pass

        # Throttle to ~max 60 fps send rate
        elapsed = loop.time() - t0
        if elapsed < 1/60:
            await asyncio.sleep(1/60 - elapsed)


# ──────────────────────────────────────────────
# STATIC FILES — MUST be mounted LAST
# ──────────────────────────────────────────────

# Evidence images (dir already created above)
app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")

# Frontend assets (css, js)
_css_dir = FRONTEND_DIR / "css"
_js_dir  = FRONTEND_DIR / "js"
_css_dir.mkdir(parents=True, exist_ok=True)
_js_dir.mkdir(parents=True, exist_ok=True)
app.mount("/css", StaticFiles(directory=str(_css_dir)), name="css")
app.mount("/js",  StaticFiles(directory=str(_js_dir)),  name="js")
