"""Test that the heartbeat thread starts, fires, and stops correctly."""

import time
import unittest
from unittest import mock

from messenger_logger.engine import LoggerEngine


UNREACHABLE = "http://localhost:19999/api/logs"


class TestHeartbeat(unittest.TestCase):
    def test_heartbeat_fires(self):
        engine = LoggerEngine(
            server_url=UNREACHABLE,
            project_name="hb_test",
            heartbeat_interval=1,
        )
        with mock.patch.object(engine, "_send_payload") as m:
            engine.start_heartbeat()
            time.sleep(2.5)
            engine.stop_heartbeat()
            heartbeat_calls = [
                c for c in m.call_args_list
                if c[0][0].get("event_type") == "heartbeat"
            ]
            self.assertGreaterEqual(len(heartbeat_calls), 1)

    def test_heartbeat_stops(self):
        engine = LoggerEngine(
            server_url=UNREACHABLE,
            project_name="hb_test",
            heartbeat_interval=1,
        )
        with mock.patch.object(engine, "_send_payload") as m:
            engine.start_heartbeat()
            time.sleep(1.5)
            engine.stop_heartbeat()
            count_after_stop = len(m.call_args_list)
            time.sleep(2)
            self.assertEqual(len(m.call_args_list), count_after_stop)

    def test_heartbeat_disabled(self):
        engine = LoggerEngine(
            server_url=UNREACHABLE,
            project_name="hb_test",
            heartbeat_interval=None,
        )
        with mock.patch.object(engine, "_send_payload") as m:
            engine.start_heartbeat()
            time.sleep(1)
            engine.stop_heartbeat()
            heartbeat_calls = [
                c for c in m.call_args_list
                if c[0][0].get("event_type") == "heartbeat"
            ]
            self.assertEqual(len(heartbeat_calls), 0)

    def test_stop_without_start(self):
        """stop_heartbeat should be safe to call even if never started."""
        engine = LoggerEngine(
            server_url=UNREACHABLE,
            project_name="hb_test",
            heartbeat_interval=60,
        )
        engine.stop_heartbeat()


if __name__ == "__main__":
    unittest.main()
