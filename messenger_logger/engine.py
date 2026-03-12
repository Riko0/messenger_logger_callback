import requests
import json
import os
import datetime
import threading
import dotenv
from typing import Dict, Any, Optional


class LoggerEngine:
    """
    Shared core for sending training events to a remote logging server.

    Handles configuration resolution, HTTP transport, ClearML auto-detection,
    heartbeat background thread, and payload construction. Used internally by
    both MessengerLogger (standalone) and MessengerLoggerCallback (HF Trainer).
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
        self._load_dotenv(dotenv_path)
        self.server_url = self._resolve_server_url(server_url)
        self.auth_token = self._resolve_auth_token(auth_token)
        self.author_username = self._resolve_author_username(author_username)
        self.metadata = self._resolve_metadata(metadata)
        self.project_name = project_name
        self.run_id = run_id or f"run_{int(datetime.datetime.now().timestamp())}"
        self.heartbeat_interval = heartbeat_interval

        self.clearml_link = self._detect_clearml_link()

        self._heartbeat_stop: Optional[threading.Event] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        print(
            f"LoggerEngine initialized for project '{self.project_name}', "
            f"run '{self.run_id}' by '{self.author_username}'"
        )
        print(f"Logs will be sent to: {self.server_url}")
        if self.clearml_link:
            print(f"ClearML task detected: {self.clearml_link}")
        if self.metadata:
            print(f"Including static metadata: {self.metadata}")

    # --- Configuration resolution ---

    def _load_dotenv(self, dotenv_path: Optional[str]):
        self.dotenv_path = dotenv_path or os.getenv("MESSENGER_LOGGER_DOTENV")
        if self.dotenv_path:
            try:
                dotenv.load_dotenv(dotenv_path=self.dotenv_path)
                print(f"Loaded environment variables from {self.dotenv_path}")
            except Exception as e:
                print(f"Warning: Could not load .env file from {self.dotenv_path}. Error: {e}")

    def _resolve_server_url(self, server_url: Optional[str]) -> str:
        url = server_url or os.getenv("MESSENGER_LOGGER_SERVER_URL")
        if not url:
            raise ValueError(
                "server_url must be provided either as an argument, via an environment "
                "variable, or within a loaded .env file."
            )
        return url

    def _resolve_auth_token(self, auth_token: Optional[str]) -> Optional[str]:
        token = auth_token or os.getenv("MESSENGER_LOGGER_AUTH_TOKEN")
        if token:
            print("Authentication token will be used for server requests.")
        return token

    def _resolve_author_username(self, author_username: Optional[str]) -> str:
        return author_username or os.getenv("MESSENGER_LOGGER_AUTHOR_USERNAME", "anonymous")

    def _resolve_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        result = metadata or {}
        env_metadata_str = os.getenv("MESSENGER_LOGGER_METADATA")
        if env_metadata_str:
            try:
                env_metadata = json.loads(env_metadata_str)
                result.update(env_metadata)
                print("Loaded metadata from environment variable.")
            except json.JSONDecodeError as e:
                print(f"Error: Could not parse MESSENGER_LOGGER_METADATA as JSON. Error: {e}")
            except Exception as e:
                print(f"Unexpected error processing metadata from env variable: {e}")
        return result

    # --- ClearML detection ---

    def _detect_clearml_link(self) -> Optional[str]:
        try:
            from clearml import Task
            task = Task.current_task()
            if task:
                return task.get_task_url()
        except ImportError:
            pass
        except Exception:
            pass
        return None

    # --- HTTP transport ---

    def send_event(
        self,
        event_type: str,
        trainer_state: Optional[Dict[str, Any]] = None,
        logs: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ):
        """Construct a full payload envelope and send it to the server."""
        payload = {
            "project_name": self.project_name,
            "run_id": self.run_id,
            "event_type": event_type,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if trainer_state is not None:
            payload["trainer_state"] = trainer_state
        if logs is not None:
            payload["logs"] = logs
        if custom_data is not None:
            payload["custom_data"] = custom_data
        if self.clearml_link:
            payload["clearml_link"] = self.clearml_link

        step = (trainer_state or {}).get("global_step")
        self._send_payload(payload, step)

    def _send_payload(self, payload: Dict[str, Any], step: Optional[int] = None):
        """Send a JSON payload to the server with error handling."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        final_payload = {
            "author_username": self.author_username,
            **self.metadata,
            **payload,
        }

        try:
            response = requests.post(
                self.server_url, json=final_payload, headers=headers, timeout=10
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            print(
                f"Warning: Request to {self.server_url} timed out for step "
                f"{step if step is not None else 'N/A'}. "
                "The server did not respond within the expected time."
            )
        except requests.exceptions.ConnectionError as e:
            print(
                f"Error: Could not connect to server at {self.server_url} for step "
                f"{step if step is not None else 'N/A'}. "
                f"The server might be unavailable or the URL is incorrect. Error details: {e}"
            )
        except requests.exceptions.HTTPError as e:
            print(
                f"Error: HTTP error occurred while sending logs for step "
                f"{step if step is not None else 'N/A'}. "
                f"Status: {e.response.status_code}, Response: {e.response.text}. "
                "Check server logs for more details."
            )
        except Exception as e:
            print(
                f"Unexpected error while sending logs for step "
                f"{step if step is not None else 'N/A'}: {e}"
            )

    # --- Heartbeat ---

    def start_heartbeat(self):
        """Start a daemon thread that sends heartbeat events at a fixed interval."""
        if not self.heartbeat_interval:
            return
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()
        print(f"Heartbeat started (interval: {self.heartbeat_interval}s)")

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(self.heartbeat_interval):
            self.send_event("heartbeat")

    def stop_heartbeat(self):
        """Stop the heartbeat background thread."""
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
            self._heartbeat_thread = None
            self._heartbeat_stop = None
