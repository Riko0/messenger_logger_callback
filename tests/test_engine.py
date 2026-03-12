"""Test LoggerEngine configuration and payload construction."""

import os
import unittest
from unittest import mock

from messenger_logger.engine import LoggerEngine


UNREACHABLE = "http://localhost:19999/api/logs"


class TestEngineConfig(unittest.TestCase):
    def test_server_url_from_arg(self):
        engine = LoggerEngine(server_url=UNREACHABLE, heartbeat_interval=None)
        self.assertEqual(engine.server_url, UNREACHABLE)

    def test_server_url_from_env(self):
        with mock.patch.dict(os.environ, {"MESSENGER_LOGGER_SERVER_URL": UNREACHABLE}):
            engine = LoggerEngine(heartbeat_interval=None)
            self.assertEqual(engine.server_url, UNREACHABLE)

    def test_missing_server_url_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                LoggerEngine(heartbeat_interval=None)

    def test_auth_token_from_arg(self):
        engine = LoggerEngine(
            server_url=UNREACHABLE, auth_token="tok123", heartbeat_interval=None,
        )
        self.assertEqual(engine.auth_token, "tok123")

    def test_auth_token_from_env(self):
        with mock.patch.dict(os.environ, {"MESSENGER_LOGGER_AUTH_TOKEN": "envtok"}):
            engine = LoggerEngine(server_url=UNREACHABLE, heartbeat_interval=None)
            self.assertEqual(engine.auth_token, "envtok")

    def test_author_defaults_to_anonymous(self):
        engine = LoggerEngine(server_url=UNREACHABLE, heartbeat_interval=None)
        self.assertEqual(engine.author_username, "anonymous")

    def test_run_id_auto_generated(self):
        engine = LoggerEngine(server_url=UNREACHABLE, heartbeat_interval=None)
        self.assertTrue(engine.run_id.startswith("run_"))

    def test_run_id_from_arg(self):
        engine = LoggerEngine(
            server_url=UNREACHABLE, run_id="my_run", heartbeat_interval=None,
        )
        self.assertEqual(engine.run_id, "my_run")

    def test_metadata_from_arg(self):
        engine = LoggerEngine(
            server_url=UNREACHABLE,
            metadata={"gpu": "A100"},
            heartbeat_interval=None,
        )
        self.assertEqual(engine.metadata, {"gpu": "A100"})

    def test_metadata_from_env(self):
        with mock.patch.dict(
            os.environ, {"MESSENGER_LOGGER_METADATA": '{"k": "v"}'}
        ):
            engine = LoggerEngine(server_url=UNREACHABLE, heartbeat_interval=None)
            self.assertIn("k", engine.metadata)

    def test_metadata_env_merge(self):
        with mock.patch.dict(
            os.environ, {"MESSENGER_LOGGER_METADATA": '{"env_key": 1}'}
        ):
            engine = LoggerEngine(
                server_url=UNREACHABLE,
                metadata={"arg_key": 2},
                heartbeat_interval=None,
            )
            self.assertIn("arg_key", engine.metadata)
            self.assertIn("env_key", engine.metadata)


class TestEnginePayload(unittest.TestCase):
    def setUp(self):
        self.engine = LoggerEngine(
            server_url=UNREACHABLE,
            project_name="p",
            run_id="r",
            auth_token="secret",
            author_username="user",
            metadata={"hw": "gpu"},
            heartbeat_interval=None,
        )

    def test_send_event_constructs_payload(self):
        with mock.patch.object(self.engine, "_send_payload") as m:
            self.engine.send_event(
                "trainer_log",
                trainer_state={"global_step": 5},
                logs={"loss": 0.1},
            )
            payload = m.call_args[0][0]
            self.assertEqual(payload["project_name"], "p")
            self.assertEqual(payload["run_id"], "r")
            self.assertEqual(payload["event_type"], "trainer_log")
            self.assertEqual(payload["logs"], {"loss": 0.1})
            self.assertEqual(payload["trainer_state"]["global_step"], 5)
            self.assertIn("timestamp", payload)

    def test_final_payload_includes_author_and_metadata(self):
        with mock.patch("messenger_logger.engine.requests.post") as m:
            m.return_value = mock.Mock(status_code=200)
            m.return_value.raise_for_status = mock.Mock()
            self.engine.send_event("heartbeat")
            call_kwargs = m.call_args
            sent = call_kwargs.kwargs["json"]
            self.assertEqual(sent["author_username"], "user")
            self.assertEqual(sent["hw"], "gpu")

    def test_auth_header_set(self):
        with mock.patch("messenger_logger.engine.requests.post") as m:
            m.return_value = mock.Mock(status_code=200)
            m.return_value.raise_for_status = mock.Mock()
            self.engine.send_event("heartbeat")
            headers = m.call_args.kwargs["headers"]
            self.assertEqual(headers["Authorization"], "Bearer secret")

    def test_clearml_link_included_when_set(self):
        self.engine.clearml_link = "https://app.clear.ml/task/123"
        with mock.patch.object(self.engine, "_send_payload") as m:
            self.engine.send_event("trainer_log")
            payload = m.call_args[0][0]
            self.assertEqual(payload["clearml_link"], "https://app.clear.ml/task/123")

    def test_clearml_link_absent_when_none(self):
        self.engine.clearml_link = None
        with mock.patch.object(self.engine, "_send_payload") as m:
            self.engine.send_event("trainer_log")
            payload = m.call_args[0][0]
            self.assertNotIn("clearml_link", payload)


if __name__ == "__main__":
    unittest.main()
