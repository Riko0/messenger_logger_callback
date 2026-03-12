from typing import Dict, Any, Optional

from .engine import LoggerEngine


class MessengerLogger:
    """
    Standalone training logger that sends events to a remote server.

    Provides simple methods mirroring the training lifecycle (start, log,
    epoch_end, finish) without requiring Hugging Face Transformers.

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
        self._state: Dict[str, Any] = {
            "global_step": 0,
            "epoch": 0.0,
            "is_training": False,
        }

    @property
    def project_name(self) -> str:
        return self._engine.project_name

    @property
    def run_id(self) -> str:
        return self._engine.run_id

    def start(self):
        """Signal the beginning of training. Starts heartbeat if enabled."""
        self._state["is_training"] = True
        self._engine.send_event("training_started", trainer_state=dict(self._state))
        self._engine.start_heartbeat()
        print(f"Training started for project '{self.project_name}', run '{self.run_id}'.")

    def log(self, step: int, metrics: Dict[str, Any], epoch: Optional[float] = None):
        """
        Log training metrics for a given step.

        Args:
            step: The current global step number.
            metrics: Dictionary of metric names to values (e.g. {"loss": 0.5}).
            epoch: Current epoch (optional, updates internal state if provided).
        """
        self._state["global_step"] = step
        if epoch is not None:
            self._state["epoch"] = epoch
        self._engine.send_event(
            "trainer_log", trainer_state=dict(self._state), logs=metrics
        )

    def epoch_end(self, epoch: int):
        """Signal the end of an epoch."""
        self._state["epoch"] = epoch
        self._engine.send_event("epoch_ended", trainer_state=dict(self._state))
        print(f"Epoch {epoch} ended for project '{self.project_name}', run '{self.run_id}'.")

    def log_custom(self, data: Dict[str, Any]):
        """
        Send arbitrary custom data to the server.

        Args:
            data: Dictionary of custom data to send.
        """
        if not isinstance(data, dict):
            print("Error: data must be a dictionary.")
            return
        self._engine.send_event("custom_log", custom_data=data)

    def finish(self):
        """Signal the end of training. Stops heartbeat."""
        self._engine.stop_heartbeat()
        self._state["is_training"] = False
        self._engine.send_event("training_finished", trainer_state=dict(self._state))
        print(f"Training finished for project '{self.project_name}', run '{self.run_id}'.")
