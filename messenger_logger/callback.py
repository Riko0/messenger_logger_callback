import dataclasses
from typing import Dict, Any, Optional

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from .engine import LoggerEngine


class MessengerLoggerCallback(TrainerCallback):
    """
    Hugging Face Trainer Callback that sends training events to a remote server.

    Intercepts Trainer lifecycle events (log, train begin/end, epoch end) and
    forwards them via HTTP. Also provides send_custom_log for arbitrary data.

    Args:
        server_url: The URL of the server endpoint. Falls back to
            MESSENGER_LOGGER_SERVER_URL env var.
        project_name: Identifier for the training project.
        run_id: Unique identifier for this run. Auto-generated if omitted.
        auth_token: Bearer token for the Authorization header. Falls back to
            MESSENGER_LOGGER_AUTH_TOKEN env var.
        author_username: Who started this run. Falls back to
            MESSENGER_LOGGER_AUTHOR_USERNAME env var, then "anonymous".
        metadata: Static key-value pairs included in every payload.
        dotenv_path: Path to a .env file to load config from.
        heartbeat_interval: Seconds between heartbeat pings (default 60).
            Set to None or 0 to disable.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        project_name: str = "default_project",
        run_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        author_username: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dotenv_path: Optional[str] = None,
        heartbeat_interval: Optional[int] = 60,
    ):
        self._engine = LoggerEngine(
            server_url=server_url,
            project_name=project_name,
            run_id=run_id,
            auth_token=auth_token,
            author_username=author_username,
            metadata=metadata,
            dotenv_path=dotenv_path,
            heartbeat_interval=heartbeat_interval,
        )

    @property
    def project_name(self) -> str:
        return self._engine.project_name

    @property
    def run_id(self) -> str:
        return self._engine.run_id

    def _get_trainer_state_info(self, state: TrainerState) -> Dict[str, Any]:
        """Extract trainer state as a plain dict, trimming log_history."""
        _state = dataclasses.asdict(state)
        log_history = _state.get("log_history", [])
        _state["log_history"] = log_history[:5]
        return _state

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            if not self._engine.clearml_link:
                self._engine.clearml_link = self._engine._detect_clearml_link()
                if self._engine.clearml_link:
                    print(f"ClearML task detected: {self._engine.clearml_link}")
            self._engine.send_event(
                "training_started",
                trainer_state=self._get_trainer_state_info(state),
            )
            self._engine.start_heartbeat()
            print(
                f"Training for project '{self.project_name}', "
                f"run '{self.run_id}' has begun."
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            self._engine.stop_heartbeat()
            self._engine.send_event(
                "training_finished",
                trainer_state=self._get_trainer_state_info(state),
            )
            print(
                f"Training for project '{self.project_name}', "
                f"run '{self.run_id}' has ended."
            )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any],
        **kwargs,
    ):
        if state.is_world_process_zero:
            self._engine.send_event(
                "trainer_log",
                trainer_state=self._get_trainer_state_info(state),
                logs=logs,
            )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            self._engine.send_event(
                "epoch_ended",
                trainer_state=self._get_trainer_state_info(state),
            )
            print(
                f"Epoch {int(state.epoch)} ended for project "
                f"'{self.project_name}', run '{self.run_id}'."
            )

    def send_custom_log(self, custom_data: Dict[str, Any]):
        """
        Send arbitrary custom data to the remote server.

        In distributed training, call only from the main process
        (check trainer.state.is_world_process_zero).
        """
        if not isinstance(custom_data, dict):
            print("Error: custom_data must be a dictionary.")
            return
        self._engine.send_event("custom_log", custom_data=custom_data)
        print(
            f"Sending custom log for project '{self.project_name}', "
            f"run '{self.run_id}'."
        )
