import json
import unittest
from unittest.mock import patch

import httpx

from app.engine.cloud import CloudFallbackEngine


class CloudFallbackEngineTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, body, *, status=200):
        self.requests = []

        def handle(request):
            self.requests.append(request)
            return httpx.Response(
                status,
                text=body,
                headers={"content-type": "text/event-stream"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        engine = CloudFallbackEngine("test-secret", base_url="https://provider.test/v1")
        with patch("app.engine.cloud.httpx.AsyncClient", return_value=client):
            return [chunk async for chunk in engine.generate_stream("hello")]

    async def test_http_errors_raise_without_exposing_provider_body(self):
        for status in (401, 403, 404, 429, 500):
            with self.subTest(status=status):
                with self.assertRaisesRegex(RuntimeError, str(status)) as error:
                    await self.collect("rejected Authorization: Bearer test-secret", status=status)
                self.assertNotIn("test-secret", str(error.exception))
                self.assertNotIn("Authorization", str(error.exception))

    async def test_usage_and_null_delta_content_are_valid(self):
        chunks = await self.collect(
            'data: {"choices": [] , "usage": {"completion_tokens": 1}}\n\n'
            'data: {"choices": [{"delta": {"role": "assistant", "content": null}}]}\n\n'
            'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            'data: {"choices": [{"delta": {"content": null}, "finish_reason": "length"}]}\n\n'
            'data: [DONE]\n\n'
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), "Hello")
        self.assertEqual(sum(chunk.is_eos for chunk in chunks), 1)
        self.assertEqual(chunks[-1].finish_reason, "length")

    async def test_sse_data_does_not_require_a_space(self):
        chunks = await self.collect(
            ': heartbeat\r\n\r\n'
            'event: message\r\n'
            'data:{"choices":[{"delta":{"content":"Hi"}}]}\r\n\r\n'
            'data:[DONE]\r\n\r\n'
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), "Hi")
        self.assertTrue(chunks[-1].is_eos)

    async def test_null_content_with_finish_reason_has_empty_text(self):
        chunks = await self.collect(
            'data: {"choices": [{"delta": {"content": null}, "finish_reason": "stop"}]}\n\n'
        )
        self.assertEqual(chunks[0].text, "")
        self.assertTrue(chunks[0].is_eos)

    async def test_transport_errors_raise_without_exposing_request_details(self):
        for failure in (httpx.ReadTimeout, httpx.ConnectError):
            with self.subTest(failure=failure.__name__):
                def handle(request):
                    raise failure("Authorization: Bearer test-secret", request=request)

                client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
                engine = CloudFallbackEngine("test-secret", base_url="https://provider.test/v1")
                with patch("app.engine.cloud.httpx.AsyncClient", return_value=client):
                    with self.assertRaises(RuntimeError) as error:
                        _ = [chunk async for chunk in engine.generate_stream("hello")]
                self.assertNotIn("test-secret", str(error.exception))

    async def test_truncated_stream_raises_instead_of_returning_partial_success(self):
        with self.assertRaisesRegex(RuntimeError, "ended.*completion"):
            await self.collect('data: {"choices": [{"delta": {"content": "partial"}}]}\n\n')

    async def test_empty_stream_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, "ended.*completion"):
            await self.collect("")

    async def test_multiline_sse_data_is_joined_before_json_parsing(self):
        chunks = await self.collect(
            'data: {"choices": [\n'
            'data: {"delta": {"content": "Hi"}, "finish_reason": "stop"}]}\n\n'
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Hi")
        self.assertTrue(chunks[0].is_eos)

    async def test_malformed_json_and_provider_errors_raise(self):
        for payload in ('{bad json', '{"error": {"message": "secret"}}', '[]',
                        '{"choices": null}', '{"choices": [{}], "missing": "delta"}',
                        '{"choices": [{"delta": {"content": ["unexpected"]}}]}'):
            with self.subTest(payload=payload):
                with self.assertRaises(RuntimeError) as error:
                    await self.collect(f"data: {payload}\n\ndata: [DONE]\n\n")
                self.assertNotIn("secret", str(error.exception))

    async def test_request_preserves_parameters_and_auth_header(self):
        await self.collect("data: [DONE]\n\n")
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://provider.test/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-secret")
        self.assertEqual(json.loads(request.content), {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
        })


if __name__ == "__main__":
    unittest.main()
