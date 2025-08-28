import asyncio
import os
from pathlib import Path

import httpx


async def main():
    # Import the app in-process
    import server

    base_dir = Path(server.BASE_DIR)
    tmp_dir = base_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sample_path = tmp_dir / "sample_single.wav"
    if not sample_path.exists():
        # Download a tiny sample from Vibe repo
        import urllib.request
        url = (
            "https://github.com/thewh1teagle/vibe/raw/main/samples/single.wav"
        )
        urllib.request.urlretrieve(url, str(sample_path))

    # Build ASGI client
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Choose first language key and a tiny local model
        lang_label = next(iter(server.LANGS.keys()))
        model_label = "Base"

        files = {"files": (sample_path.name, sample_path.open("rb"), "audio/wav")}
        data = {
            "use_api": "0",
            "api_key": "",
            "model_label": model_label,
            "lang_label": lang_label,
        }

        resp = await client.post("/api/transcribe", files=files, data=data)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        print("Pretest job:", job_id)

        # Poll a few times to ensure it starts properly
        for _ in range(20):
            status = await client.get(f"/api/status/{job_id}")
            status.raise_for_status()
            dat = status.json()
            print("status:", dat.get("status"), "progress:", dat.get("progress"))
            if dat.get("status") in {"done", "error"}:
                break
            await asyncio.sleep(1)

        # Final output
        status = await client.get(f"/api/status/{job_id}")
        status.raise_for_status()
        dat = status.json()
        print("final:", dat.get("status"), "progress:", dat.get("progress"))
        # Print last few logs
        logs = dat.get("logs") or []
        tail = "\n".join(logs[-10:])
        try:
            print(tail)
        except Exception:
            # Fallback for Windows console encoding
            print(tail.encode("cp1252", errors="ignore").decode("cp1252", errors="ignore"))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
