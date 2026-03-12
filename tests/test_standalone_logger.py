"""Test the standalone MessengerLogger against an unreachable server.

Every method must complete without raising, printing connection errors
to stdout instead of crashing the training process.
"""

import unittest
from unittest import mock

from messenger_logger import MessengerLogger


UNREACHABLE = "http://localhost:19999/api/logs"


class TestStandaloneLogger(unittest.TestCase):
    def setUp(self):
        self.logger = MessengerLogger(
            server_url=UNREACHABLE,
            project_name="test_project",
            run_id="test_run",
            author_username="tester",
            heartbeat_interval=None,
        )

    def test_properties(self):
        self.assertEqual(self.logger.project_name, "test_project")
        self.assertEqual(self.logger.run_id, "test_run")

    def test_start_does_not_crash(self):
        self.logger.start()

    def test_log_does_not_crash(self):
        self.logger.start()
        self.logger.log(step=1, metrics={"loss": 0.5}, epoch=0.1)

    def test_epoch_end_does_not_crash(self):
        self.logger.start()
        self.logger.epoch_end(epoch=1)

    def test_log_custom_does_not_crash(self):
        self.logger.log_custom({"key": "value"})

    def test_log_custom_rejects_non_dict(self):
        self.logger.log_custom("not a dict")

    def test_finish_does_not_crash(self):
        self.logger.start()
        self.logger.finish()

    def test_full_lifecycle(self):
        self.logger.start()
        self.logger.log(step=1, metrics={"loss": 0.9}, epoch=0.0)
        self.logger.log(step=2, metrics={"loss": 0.7}, epoch=0.5)
        self.logger.epoch_end(epoch=1)
        self.logger.log_custom({"checkpoint": "/tmp/best"})
        self.logger.finish()

    def test_send_event_payload(self):
        """Verify the payload structure passed to _send_payload."""
        with mock.patch.object(self.logger._engine, "_send_payload") as m:
            self.logger.start()
            call_args = m.call_args[0][0]
            self.assertEqual(call_args["event_type"], "training_started")
            self.assertEqual(call_args["project_name"], "test_project")
            self.assertEqual(call_args["run_id"], "test_run")
            self.assertIn("trainer_state", call_args)
            self.assertTrue(call_args["trainer_state"]["is_training"])

    def test_log_payload_contains_metrics(self):
        with mock.patch.object(self.logger._engine, "_send_payload") as m:
            self.logger.log(step=5, metrics={"loss": 0.3, "lr": 1e-4}, epoch=1.0)
            call_args = m.call_args[0][0]
            self.assertEqual(call_args["event_type"], "trainer_log")
            self.assertEqual(call_args["logs"], {"loss": 0.3, "lr": 1e-4})
            self.assertEqual(call_args["trainer_state"]["global_step"], 5)
            self.assertEqual(call_args["trainer_state"]["epoch"], 1.0)


if __name__ == "__main__":
    unittest.main()
