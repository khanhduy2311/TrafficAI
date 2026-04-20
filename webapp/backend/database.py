"""
Database module — SQLite async via aiosqlite
Bảng violations: lưu toàn bộ vi phạm phát hiện được
"""

import aiosqlite
import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "violations.db"

EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence"


async def init_db():
    """Tạo bảng nếu chưa có."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id        INTEGER,
                vehicle_type    TEXT,
                violation_type  TEXT NOT NULL,
                confidence      REAL,
                timestamp       TEXT NOT NULL,
                frame_number    INTEGER,
                bbox            TEXT,
                evidence_path   TEXT,
                session_id      TEXT
            )
        """)
        await db.commit()


async def add_violation(
    track_id: int,
    vehicle_type: str,
    violation_type: str,
    confidence: float,
    frame_number: int,
    bbox: list,
    evidence_path: str = None,
    session_id: str = None,
):
    """Thêm 1 vi phạm vào DB, trả về ID."""
    ts = datetime.now().isoformat()
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute(
            """INSERT INTO violations
               (track_id, vehicle_type, violation_type, confidence,
                timestamp, frame_number, bbox, evidence_path, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                track_id,
                vehicle_type,
                violation_type,
                round(confidence, 4),
                ts,
                frame_number,
                json.dumps(bbox),
                evidence_path,
                session_id,
            ),
        )
        await db.commit()
        return cursor.lastrowid


async def get_violations(limit: int = 200, offset: int = 0):
    """Lấy danh sách vi phạm, mới nhất trước."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM violations ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_violation_count():
    """Đếm tổng vi phạm."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM violations")
        row = await cursor.fetchone()
        return row[0]


async def get_stats():
    """Thống kê vi phạm theo loại."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT violation_type, COUNT(*) as count
               FROM violations GROUP BY violation_type"""
        )
        rows = await cursor.fetchall()
        by_type = {r["violation_type"]: r["count"] for r in rows}

        cursor2 = await db.execute("SELECT COUNT(*) FROM violations")
        total = (await cursor2.fetchone())[0]

        return {"total": total, "by_type": by_type}


async def clear_violations():
    """Xoá toàn bộ vi phạm."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("DELETE FROM violations")
        await db.commit()
    # Xoá ảnh bằng chứng
    if EVIDENCE_DIR.exists():
        for f in EVIDENCE_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)


async def export_csv() -> str:
    """Xuất toàn bộ vi phạm ra CSV string."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM violations ORDER BY id DESC"
        )
        rows = await cursor.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Track ID", "Vehicle Type", "Violation Type",
        "Confidence", "Timestamp", "Frame", "BBox", "Evidence Path",
    ])
    for r in rows:
        writer.writerow([
            r["id"], r["track_id"], r["vehicle_type"],
            r["violation_type"], r["confidence"], r["timestamp"],
            r["frame_number"], r["bbox"], r["evidence_path"],
        ])
    return output.getvalue()
