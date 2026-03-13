import dataclasses
from typing import Dict, Any, Optional

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from .engine import LoggerEngine


class MessengerLoggerCallback(TrainerCallback):
    """
    Hugging Face Trainer Callback that sends training events to a remote server.

    Safe by design — if the server is unreachable or the URL is not configured,
    all methods silently do nothing. The callback will never raise an exception
    or interfere with training.

    Args:
        server_url: The URL of the server endpoint. Falls back to
            MESSENGER_LOGGER_SERVER_URL env var. If neither is set the
            callback becomes a no-op.
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
        clearml_link: Explicit ClearML task URL. If not provided, the
            logger attempts auto-detection via Task.current_task() and
            CLEARML_TASK_ID env var.
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
        clearml_link: Optional[str] = None,
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
            clearml_link=clearml_link,
        )

    @property
    def project_name(self) -> str:
        return getattr(self._engine, "project_name", "default_project")

    @property
    def run_id(self) -> str:
        return getattr(self._engine, "run_id", "")

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
            self._engine.send_event(
                "training_started",
                trainer_state=self._get_trainer_state_info(state),
            )
            self._engine.start_heartbeat()

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

    def send_custom_log(self, custom_data: Dict[str, Any]):
        """
        Send arbitrary custom data to the remote server.

        In distributed training, call only from the main process
        (check trainer.state.is_world_process_zero).
        """
        if not isinstance(custom_data, dict):
            return
        self._engine.send_event("custom_log", custom_data=custom_data)
