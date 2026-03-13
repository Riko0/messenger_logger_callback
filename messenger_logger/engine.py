import requests
import json
import os
import datetime
import threading
import traceback
import dotenv
from typing import Dict, Any, Optional


def _safe(method):
    """Decorator that swallows all exceptions so the logger never crashes training."""
    def wrapper(self, *args, **kwargs):
        if not self._active:
            return
        try:
            return method(self, *args, **kwargs)
        except Exception:
            traceback.print_exc()
            print(f"Warning: MessengerLogger.{method.__name__}() failed (see above). Training continues.")
    return wrapper


class LoggerEngine:
    """
    Shared core for sending training events to a remote logging server.

    Handles configuration resolution, HTTP transport, ClearML auto-detection,
    heartbeat background thread, and payload construction. Used internally by
    both MessengerLogger (standalone) and MessengerLoggerCallback (HF Trainer).

    If ``server_url`` is not provided (and not in env), the engine becomes a
    silent no-op — no exceptions, no network calls. Same when ``rank`` is set
    to any value other than 0.
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
        rank: Optional[int] = None,
        clearml_link: Optional[str] = None,
    ):
        self._active = False
        self._heartbeat_stop: Optional[threading.Event] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        if rank is not None and rank != 0:
            return

        try:
            self._load_dotenv(dotenv_path)
            self.server_url = self._resolve_server_url(server_url)
        except Exception:
            return

        if not self.server_url:
            return

        self._active = True
        self.auth_token = self._resolve_auth_token(auth_token)
        self.author_username = self._resolve_author_username(author_username)
        self.metadata = self._resolve_metadata(metadata)
        self.project_name = project_name
        self.run_id = run_id or f"run_{int(datetime.datetime.now().timestamp())}"
        self.heartbeat_interval = heartbeat_interval

        self.clearml_link: Optional[str] = clearml_link or self._detect_clearml_link()

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

    def _resolve_server_url(self, server_url: Optional[str]) -> Optional[str]:
        url = server_url or os.getenv("MESSENGER_LOGGER_SERVER_URL")
        if not url:
            print(
                "MessengerLogger: server_url not provided and MESSENGER_LOGGER_SERVER_URL "
                "not set. Logger will be inactive (no-op)."
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

    # --- ClearML detection (lazy — retries until found) ---

    def _detect_clearml_link(self) -> Optional[str]:
        # 1. Try Task.current_task() (works if Task.init() was called in this process)
        try:
            from clearml import Task
            task = Task.current_task()
            if task:
                print(f"[MessengerLogger] ClearML task found: id={task.id}")
                url = task.get_task_url()
                if url:
                    print(f"[MessengerLogger] ClearML URL: {url}")
                    return url
                # get_task_url() returned empty — build URL manually
                web_host = os.getenv("CLEARML_WEB_HOST", "").rstrip("/")
                if not web_host:
                    api_host = os.getenv("CLEARML_API_HOST", "")
                    if api_host:
                        web_host = api_host.replace("://api.", "://app.")
                if web_host:
                    project_id = task.get_project_name() or "*"
                    url = f"{web_host}/projects/*/experiments/{task.id}/output/log"
                    print(f"[MessengerLogger] ClearML URL (built from env): {url}")
                    return url
                print(f"[MessengerLogger] ClearML task found but could not build URL")
            else:
                print("[MessengerLogger] ClearML imported but Task.current_task() is None")
        except ImportError:
            print("[MessengerLogger] ClearML not installed")
            return None
        except Exception as e:
            print(f"[MessengerLogger] ClearML detection error: {e}")

        # 2. Try building the URL from CLEARML_TASK_ID env var
        task_id = os.getenv("CLEARML_TASK_ID")
        if task_id:
            print(f"[MessengerLogger] Found CLEARML_TASK_ID={task_id}")
            api_host = os.getenv("CLEARML_API_HOST", "")
            if api_host:
                web_host = api_host.replace("://api.", "://app.").rstrip("/")
            else:
                web_host = os.getenv("CLEARML_WEB_HOST", "").rstrip("/")
            if web_host:
                url = f"{web_host}/projects/*/experiments/{task_id}/output/log"
                print(f"[MessengerLogger] ClearML URL (from env): {url}")
                return url
            try:
                from clearml import Task
                task = Task.get_task(task_id=task_id)
                if task:
                    url = task.get_task_url()
                    if url:
                        return url
            except Exception as e:
                print(f"[MessengerLogger] ClearML get_task error: {e}")

        return None

    def _ensure_clearml_link(self):
        """Re-check ClearML if not found yet. Called lazily on send_event."""
        if self.clearml_link is not None:
            return
        link = self._detect_clearml_link()
        if link:
            self.clearml_link = link
            print(f"ClearML task detected (lazy): {self.clearml_link}")

    # --- HTTP transport ---

    @_safe
    def send_event(
        self,
        event_type: str,
        trainer_state: Optional[Dict[str, Any]] = None,
        logs: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ):
        """Construct a full payload envelope and send it to the server."""
        self._ensure_clearml_link()

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

    @_safe
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

    @_safe
    def stop_heartbeat(self):
        """Stop the heartbeat background thread."""
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
            self._heartbeat_thread = None
            self._heartbeat_stop = None
