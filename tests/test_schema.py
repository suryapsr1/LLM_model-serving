import unittest

from pydantic import ValidationError

from app.schemas.inference import ChatCompletionRequest


class ChatCompletionRequestTests(unittest.TestCase):
    def test_defaults_and_supported_engines(self):
        request = ChatCompletionRequest(prompt="Hello")
        self.assertEqual(request.engine, "local")
        self.assertEqual(request.max_tokens, 128)
        self.assertEqual(request.temperature, 0.7)
        self.assertEqual(request.top_p, 0.9)
        self.assertEqual(
            ChatCompletionRequest(prompt="Hello", engine="cloud").engine, "cloud"
        )

    def test_explicit_null_values_are_rejected(self):
        for field in ("max_tokens", "temperature", "top_p", "engine"):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    ChatCompletionRequest.model_validate({"prompt": "Hello", field: None})

    def test_unknown_engines_are_rejected(self):
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(prompt="Hello", engine="missing")

    def test_generation_limits_are_validated(self):
        for values in (
            {"prompt": ""},
            {"max_tokens": 0},
            {"max_tokens": 2049},
            {"temperature": -0.1},
            {"temperature": 2.1},
            {"top_p": -0.1},
            {"top_p": 1.1},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    ChatCompletionRequest.model_validate({"prompt": "Hello", **values})


if __name__ == "__main__":
    unittest.main()
