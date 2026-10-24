"""Offline tests for the pinned fixture fetcher's retry policy."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import io
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

import fetch_photo_fixture as fixture


class FetchTests(unittest.TestCase):
    def error(self, status, retry=None):
        headers = {} if retry is None else {"Retry-After": retry}
        return urllib.error.HTTPError("https://example.invalid/pinned", status, "busy", headers, io.BytesIO())

    def test_bounded_retries_grow_and_keep_the_same_request(self):
        errors = [self.error(429, "45"), self.error(503), self.error(429, "2"), self.error(429)]
        with patch.object(fixture.urllib.request, "urlopen", side_effect=errors) as opened, patch.object(fixture.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 3 retries"):
                fixture.fetch("https://example.invalid/pinned")
        self.assertEqual(opened.call_count, 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 45, 1, 90, 1, 180, 1])
        requests = [call.args[0] for call in opened.call_args_list]
        self.assertTrue(all(request is requests[0] for request in requests))
        self.assertIn("https://github.com/CertaMesh/seyk", requests[0].get_header("User-agent"))

    def test_success_after_retry_and_nonretryable_failure(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"pinned bytes"
        with patch.object(fixture.urllib.request, "urlopen", side_effect=[self.error(503), response]), patch.object(fixture.time, "sleep"):
            self.assertEqual(fixture.fetch("https://example.invalid/pinned"), b"pinned bytes")
        with patch.object(fixture.urllib.request, "urlopen", side_effect=self.error(404)) as opened, patch.object(fixture.time, "sleep"):
            with self.assertRaises(urllib.error.HTTPError):
                fixture.fetch("https://example.invalid/pinned")
            self.assertEqual(opened.call_count, 1)

    def test_retry_after_accepts_seconds_dates_and_bad_headers(self):
        self.assertEqual(fixture.retry_after("120"), 120)
        self.assertEqual(fixture.retry_after(None), 30)
        self.assertEqual(fixture.retry_after("bad"), 30)
        self.assertEqual(fixture.retry_after("inf"), 30)
        self.assertEqual(fixture.retry_after("nan"), 30)
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120))
        self.assertTrue(118 <= fixture.retry_after(future) <= 120)


if __name__ == "__main__":
    unittest.main()
