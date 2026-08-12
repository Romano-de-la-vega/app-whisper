import asyncio
import wave
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
        # Échantillon silencieux local : le smoke test ne dépend pas du réseau.
        with wave.open(str(sample_path), "wb") as sample:
            sample.setnchannels(1)
            sample.setsampwidth(2)
            sample.setframerate(16_000)
            sample.writeframes(b"\x00\x00" * 16_000)

    # Build ASGI client
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        # Choose first language key and a tiny local model
        lang_label = next(iter(server.LANGS.keys()))
        model_label = "Small"

        data = {
            "use_api": "0",
            "api_key": "",
            "model_label": model_label,
            "lang_label": lang_label,
        }

        with sample_path.open("rb") as sample_file:
            files = {"files": (sample_path.name, sample_file, "audio/wav")}
            resp = await client.post(
                "/api/transcribe",
                files=files,
                data=data,
                headers={"X-Whisper-CSRF": server.CSRF_TOKEN},
            )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        print("Pretest job:", job_id)

        # Attendre la fin réelle du worker local.
        dat = {}
        for _ in range(600):
            status = await client.get(f"/api/status/{job_id}")
            status.raise_for_status()
            dat = status.json()
            print("status:", dat.get("status"), "progress:", dat.get("progress"))
            if dat.get("status") in {"done", "error"}:
                break
            await asyncio.sleep(0.2)
        else:
            raise TimeoutError("Le smoke test n'est pas terminé après 120 secondes.")

        # Final output
        status = await client.get(f"/api/status/{job_id}")
        status.raise_for_status()
        dat = status.json()
        print("final:", dat.get("status"), "progress:", dat.get("progress"))
        if dat.get("status") != "done":
            raise RuntimeError(f"Smoke test en échec : {dat.get('logs', [])[-5:]}")
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
