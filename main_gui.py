import threading, time, socket, uvicorn, webview
from server import app  # 👈 import direct

def is_up(h, p):
    s = socket.socket(); s.settimeout(0.5)
    try: s.connect((h, p)); return True
    except OSError: return False
    finally: s.close()

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="warning", workers=1)

if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    for _ in range(60):
        if is_up("127.0.0.1", 8000): break
        time.sleep(0.5)
    webview.create_window("Transcripteur Whisper", "http://127.0.0.1:8000", width=1100, height=740)
    webview.start()
