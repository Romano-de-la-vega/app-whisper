import json
import socket
import threading
import time
from urllib.request import urlopen

import uvicorn
import webview

from server import app, shutdown_background_services


def _is_our_server_ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
            payload = json.load(response)
        return payload.get("service") == "transcripteur-whisper" and bool(payload.get("ok"))
    except Exception:
        return False


if __name__ == "__main__":
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(("127.0.0.1", 0))
    listen_socket.listen(128)
    port = listen_socket.getsockname()[1]

    config = uvicorn.Config(app, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listen_socket]},
        daemon=True,
        name="whisper-web-server",
    )
    server_thread.start()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and server_thread.is_alive():
        if _is_our_server_ready(port):
            break
        time.sleep(0.2)
    else:
        server.should_exit = True
        listen_socket.close()
        raise RuntimeError("Le serveur local du transcripteur n'a pas pu démarrer.")

    webview.create_window(
        "Transcripteur Whisper",
        f"http://127.0.0.1:{port}",
        width=1100,
        height=740,
    )
    try:
        webview.start()
    finally:
        server.should_exit = True
        shutdown_background_services(wait=False)
        server_thread.join(timeout=5)
        listen_socket.close()
