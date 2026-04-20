"""
Traffic Violation Detection System — Startup Script
----------------------------------------------------
Chạy lệnh này để khởi động toàn bộ hệ thống:
    python run.py

Sau đó mở trình duyệt tại: http://localhost:8000
KHÔNG cần chạy frontend riêng — backend đã serve sẵn frontend.
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 55)
    print("  TrafficAI — Traffic Violation Detection System")
    print("=" * 55)
    print("  ► Backend + Frontend served at: http://localhost:5000")
    print("  ► WebSocket:                    ws://localhost:5000/ws/video")
    print("  ► API Docs:                     http://localhost:5000/docs")
    print("=" * 55)
    print("  [!] Đừng mở trực tiếp file index.html trên trình duyệt")
    print("  [!] Truy cập qua http://localhost:5000 sau khi khởi động")
    print("=" * 55)

    uvicorn.run(
        "webapp.backend.main:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info",
    )
