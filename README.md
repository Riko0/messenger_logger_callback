# Messenger Logger Callback

A custom Hugging Face Trainer Callback for sending training logs and custom data to a remote server with authentication.

## Overview

`messenger-logger-callback` is a Python library designed to easily integrate remote logging into your Hugging Face Trainer workflows. It provides a `TrainerCallback` that automatically captures standard training metrics (loss, learning rate, epoch, etc.) and sends them as JSON payloads to a specified HTTP endpoint. Additionally, it offers a flexible method to send arbitrary custom data from anywhere in your application.

This library is particularly useful for:

* Centralized logging of machine learning experiments.
* Real-time monitoring of training progress on a remote dashboard.
* Integrating with custom notification systems (e.g., Telegram bots, Slack webhooks) by having a server endpoint process the received logs.

## Features

* **Hugging Face Trainer Integration**: Seamlessly plugs into the Hugging Face `Trainer` class.
* **Automatic Log Capture**: Intercepts `on_log`, `on_train_begin`, `on_train_end`, and `on_epoch_end` events.
* **Custom Log Sending**: Provides a `send_custom_log` method for sending any arbitrary JSON data.
* **Flexible Configuration**: Server URL and authentication token can be provided via constructor arguments or environment variables.
* **Robust Error Handling**: Includes try-except blocks for network requests to gracefully handle timeouts, connection errors, and HTTP errors, printing informative messages without crashing your training.
* **Authentication Support**: Supports sending a Bearer token in the `Authorization` header for secure communication with your logging server.

## Installation

You can install `messenger-logger-callback` using pip:

```bash
pip install messenger-logger-callback
```

## Usage

### 1. Basic Integration with Hugging Face Trainer

```Python
from transformers import Trainer, TrainingArguments
from messenger_logger.callback import MessengerLoggerCallback
import os

# --- Configure your server URL and optional authentication token, username, and metadata ---
# Option A: Pass directly to the constructor
messenger_logger = MessengerLoggerCallback(
    server_url="http://your-logging-server.com/api/logs",
    project_name="my_awesome_model",
    run_id="experiment_v2",
    auth_token="your_secret_api_token",
    author_username="john.doe",
    metadata={"framework": "pytorch", "model_type": "bert-large"}
)

# Option B: Set as environment variables (recommended for production)
# os.environ["MESSENGER_LOGGER_SERVER_URL"] = "http://your-logging-server.com/api/logs"
# os.environ["MESSENGER_LOGGER_AUTH_TOKEN"] = "your_secret_api_token"
# os.environ["MESSENGER_LOGGER_AUTHOR_USERNAME"] = "jane.smith"
# os.environ["MESSENGER_LOGGER_METADATA"] = '{"dataset": "squad", "batch_size": 32}'

# If using environment variables, you can omit the arguments:
# messenger_logger = MessengerLoggerCallback(
#     project_name="my_awesome_model",
#     run_id="experiment_v2"
# )

...

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    callbacks=[messenger_logger] # Add your custom callback here
)
```

### 2. Sending Custom Logs
You can send arbitrary data at any point using the send_custom_log method:

```Python
from messenger_logger.callback import MessengerLoggerCallback
import os

# Ensure the logger is initialized (e.g., from environment variables)
# os.environ["MESSENGER_LOGGER_SERVER_URL"] = "http://localhost:5000/api/logs"
# os.environ["MESSENGER_LOGGER_AUTH_TOKEN"] = "my_custom_token"
# os.environ["MESSENGER_LOGGER_AUTHOR_USERNAME"] = "jane.smith"
custom_logger = MessengerLoggerCallback(
    server_url="http://localhost:5000/api/logs", # Or omit if using env vars
    project_name="my_inference_project",
    run_id="prediction_run_1"
)

# Send custom data, e.g., after model evaluation or deployment
custom_logger.send_custom_log({
    "event": "model_evaluation_complete",
    "model_version": "v1.2.0",
    "evaluation_metrics": {
        "accuracy": 0.92,
        "f1_score": 0.915,
        "precision": 0.90,
        "recall": 0.93
    },
    "dataset_info": "test_set_2023-01-15"
})

custom_logger.send_custom_log({
    "event": "alert",
    "level": "CRITICAL",
    "message": "High GPU temperature detected on node gpu-01",
    "temperature_celsius": 85,
    "timestamp": "2023-10-27T10:30:00Z"
})
```

## Configuration

### MessengerLoggerCallback can be configured using a three-tiered precedence system:

Constructor Arguments: Take the highest precedence.

Environment Variables: Overridden by constructor arguments.

.env File: Loaded if dotenv_path is provided, and overridden by both environment variables and constructor arguments.

### The available arguments and corresponding environment variables are:

| Constructor Argument | Environment Variable | Description |
| -----| ----- | ----- |
| server_url | MESSENGER_LOGGER_SERVER_URL | The HTTP endpoint to send logs to. Required. |
| project_name | (n/a) | A string identifier for your project. Defaults to "default_project". |
| run_id | (n/a) | A unique identifier for the current training run. If not provided, a timestamp-based ID is generated. |
| auth_token | MESSENGER_LOGGER_AUTH_TOKEN | An authentication token for the Authorization header. |
| author_username | MESSENGER_LOGGER_AUTHOR_USERNAME | The username of the author. Defaults to "anonymous". |
| metadata | MESSENGER_LOGGER_METADATA | A dictionary of static metadata. The environment variable should be a valid JSON string
| dotenv_path | MESSENGER_LOGGER_DOTENV | Path to a .env file to load variables from. |


## Error Handling
The library includes robust error handling for network requests. If the logging server is unavailable, times out, or returns an HTTP error (4xx/5xx), a warning or error message will be printed to the console, but your training script will continue to run without interruption.

Example error messages you might see:

``
Warning: Request to http://localhost:5000/api/logs timed out for step 10. The server did not respond within the expected time.

Error: Could not connect to server at http://localhost:9999/api/logs for step N/A. The server might be unavailable or the URL is incorrect. Error details: ...

Error: HTTP error occurred while sending logs for step 20. Status: 401, Response: Unauthorized. Check server logs for more details.
``

## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests on the GitHub repository.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
