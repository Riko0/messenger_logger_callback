"""Test the MessengerLoggerCallback against an unreachable server.

All Trainer callback hooks must complete without raising.
"""

import unittest
from unittest import mock

from transformers import TrainingArguments, TrainerState, TrainerControl
from messenger_logger.callback import MessengerLoggerCallback


UNREACHABLE = "http://localhost:19999/api/logs"


def _make_state(**overrides):
    state = TrainerState()
    state.global_step = overrides.get("global_step", 10)
    state.epoch = overrides.get("epoch", 0.5)
    state.is_training = overrides.get("is_training", True)
    state.is_world_process_zero = overrides.get("is_world_process_zero", True)
    return state


class TestCallback(unittest.TestCase):
    def setUp(self):
        self.cb = MessengerLoggerCallback(
            server_url=UNREACHABLE,
            project_name="cb_project",
            run_id="cb_run",
            heartbeat_interval=None,
        )
        self.args = TrainingArguments(output_dir="/tmp/test_cb")
        self.state = _make_state()
        self.control = TrainerControl()

    def test_properties(self):
        self.assertEqual(self.cb.project_name, "cb_project")
        self.assertEqual(self.cb.run_id, "cb_run")

    def test_on_train_begin(self):
        self.cb.on_train_begin(self.args, self.state, self.control)

    def test_on_log(self):
        self.cb.on_log(self.args, self.state, self.control, logs={"loss": 0.3})

    def test_on_epoch_end(self):
        self.cb.on_epoch_end(self.args, self.state, self.control)

    def test_on_train_end(self):
        self.cb.on_train_end(self.args, self.state, self.control)

    def test_send_custom_log(self):
        self.cb.send_custom_log({"key": "value"})

    def test_send_custom_log_rejects_non_dict(self):
        self.cb.send_custom_log("not a dict")

    def test_skips_non_zero_process(self):
        """Events from non-zero processes should be silently skipped."""
        state = _make_state(is_world_process_zero=False)
        with mock.patch.object(self.cb._engine, "send_event") as m:
            self.cb.on_log(self.args, state, self.control, logs={"loss": 0.1})
            m.assert_not_called()

    def test_full_lifecycle(self):
        self.cb.on_train_begin(self.args, self.state, self.control)
        self.cb.on_log(self.args, self.state, self.control, logs={"loss": 0.5})
        self.cb.on_epoch_end(self.args, self.state, self.control)
        self.cb.send_custom_log({"info": "test"})
        self.cb.on_train_end(self.args, self.state, self.control)

    def test_on_log_payload(self):
        with mock.patch.object(self.cb._engine, "_send_payload") as m:
            self.cb.on_log(self.args, self.state, self.control, logs={"loss": 0.2})
            payload = m.call_args[0][0]
            self.assertEqual(payload["event_type"], "trainer_log")
            self.assertEqual(payload["logs"], {"loss": 0.2})
            self.assertIn("trainer_state", payload)


if __name__ == "__main__":
    unittest.main()
