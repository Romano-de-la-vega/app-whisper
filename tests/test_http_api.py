from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

import server


class HttpApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://127.0.0.1",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_health_identifies_the_application_and_mp3_encoder(self) -> None:
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service"], "transcripteur-whisper")
        self.assertTrue(response.json()["mp3_encoder"])

    async def test_home_page_and_static_assets_render(self) -> None:
        page = await self.client.get("/")
        script = await self.client.get("/static/app.js")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Conversion MP4 automatique", page.text)
        self.assertEqual(script.status_code, 200)
        self.assertIn("requestCancellation", script.text)

    async def test_invalid_extension_is_rejected_without_creating_a_job(self) -> None:
        response = await self.client.post(
            "/api/transcribe",
            data={
                "use_api": "0",
                "model_label": "Small",
                "lang_label": "Français",
            },
            files={"files": ("danger.exe", b"not-media", "application/octet-stream")},
            headers={"X-Whisper-CSRF": server.CSRF_TOKEN},
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("Format non pris en charge", response.json()["detail"])

    async def test_oversized_content_length_is_rejected_before_form_parsing(self) -> None:
        response = await self.client.post(
            "/api/transcribe",
            content=b"",
            headers={
                "content-length": str(server.REQUEST_BODY_LIMIT_BYTES + 1),
                "X-Whisper-CSRF": server.CSRF_TOKEN,
            },
        )
        self.assertEqual(response.status_code, 413)

    async def test_csrf_middleware_rejects_missing_token_and_accepts_server_token(self) -> None:
        missing_job_id = "f" * 32

        rejected = await self.client.post(f"/api/cancel/{missing_job_id}")
        accepted = await self.client.post(
            f"/api/cancel/{missing_job_id}",
            headers={"X-Whisper-CSRF": server.CSRF_TOKEN},
        )

        self.assertEqual(rejected.status_code, 403)
        self.assertEqual(accepted.status_code, 404)

    async def test_chunked_body_without_content_length_is_bounded(self) -> None:
        boundary = "whisper-test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="use_api"\r\n\r\n'
            "0\r\n"
            f"--{boundary}--\r\n"
        ).encode("ascii")

        async def oversized_stream():
            yield body[:20]
            yield body[20:]

        with patch.object(server, "REQUEST_BODY_LIMIT_BYTES", 32):
            response = await self.client.post(
                "/api/transcribe",
                content=oversized_stream(),
                headers={
                    "content-type": f"multipart/form-data; boundary={boundary}",
                    "X-Whisper-CSRF": server.CSRF_TOKEN,
                },
            )

        self.assertNotIn("content-length", response.request.headers)
        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
