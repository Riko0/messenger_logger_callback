Messenger Logger Callback
A custom Hugging Face Trainer Callback for sending training logs and custom data to a remote server with authentication.

Overview
messenger-logger-callback is a Python library designed to easily integrate remote logging into your Hugging Face Trainer workflows. It provides a TrainerCallback that automatically captures standard training metrics (loss, learning rate, epoch, etc.) and sends them as JSON payloads to a specified HTTP endpoint. Additionally, it offers a flexible method to send arbitrary custom data from anywhere in your application.

This library is particularly useful for:

Centralized logging of machine learning experiments.

Real-time monitoring of training progress on a remote dashboard.

Integrating with custom notification systems (e.g., Telegram bots, Slack webhooks) by having a server endpoint process the received logs.

Features
Hugging Face Trainer Integration: Seamlessly plugs into the Hugging Face Trainer class.

Automatic Log Capture: Intercepts on_log, on_train_begin, on_train_end, and on_epoch_end events.

Custom Log Sending: Provides a send_custom_log method for sending any arbitrary JSON data.

Flexible Configuration: Server URL and authentication token can be provided via constructor arguments or environment variables.

Robust Error Handling: Includes try-except blocks for network requests to gracefully handle timeouts, connection errors, and HTTP errors, printing informative messages without crashing your training.

Authentication Support: Supports sending a Bearer token in the Authorization header for secure communication with your logging server.

Installation
You can install messenger-logger-callback using pip:

pip install messenger-logger-callback

Usage
1. Basic Integration with Hugging Face Trainer
from transformers import Trainer, TrainingArguments
from messenger_logger.callback import MessengerLoggerCallback # Assuming your file is in messenger_logger/callback.py
import os

# --- Configure your server URL and optional authentication token ---
# Option A: Pass directly to the constructor
SERVER_URL = "http://your-logging-server.com/api/logs"
AUTH_TOKEN = "your_secret_api_token"

# Option B: Set as environment variables (recommended for production)
# os.environ["MESSENGER_LOGGER_SERVER_URL"] = "http://your-logging-server.com/api/logs"
# os.environ["MESSENGER_LOGGER_AUTH_TOKEN"] = "your_secret_api_token"

# Initialize the callback
# If using environment variables, you can omit server_url and auth_token arguments:
# messenger_logger = MessengerLoggerCallback(
#     project_name="my_awesome_model",
#     run_id="experiment_v2"
# )
messenger_logger = MessengerLoggerCallback(
    server_url=SERVER_URL,
    project_name="my_awesome_model",
    run_id="experiment_v2",
    auth_token=AUTH_TOKEN
)

# Define your TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=100,
    report_to=["tensorboard"] # You can still report to other services
)

# Initialize your Trainer (replace with your actual model and dataset)
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from datasets import load_dataset
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# dataset = load_dataset("imdb")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     callbacks=[messenger_logger] # Add your custom callback here
# )

# For demonstration purposes, let's create a dummy Trainer
class DummyModel:
    pass
class DummyDataset:
    def __len__(self): return 1000
    def __getitem__(self, idx): return {"input_ids": [1,2,3], "labels": 0}

trainer = Trainer(
    model=DummyModel(),
    args=training_args,
    train_dataset=DummyDataset(),
    callbacks=[messenger_logger] # Add your custom callback here
)

# Start training
print("Starting dummy training...")
trainer.train()
print("Dummy training finished.")

2. Sending Custom Logs
You can send arbitrary data at any point using the send_custom_log method:

from messenger_logger.callback import MessengerLoggerCallback
import os

# Ensure the logger is initialized (e.g., from environment variables)
# os.environ["MESSENGER_LOGGER_SERVER_URL"] = "http://localhost:5000/api/logs"
# os.environ["MESSENGER_LOGGER_AUTH_TOKEN"] = "my_custom_token"
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

Configuration
The MessengerLoggerCallback can be configured using:

Constructor Arguments:

server_url (str, optional): The HTTP endpoint to send logs to.

project_name (str, optional): A string identifier for your project (defaults to "default_project").

run_id (str, optional): A unique identifier for the current training run. If not provided, a timestamp-based ID is generated.

auth_token (str, optional): An authentication token to include in the Authorization: Bearer <token> header.

Environment Variables:

MESSENGER_LOGGER_SERVER_URL: Overrides server_url if set.

MESSENGER_LOGGER_AUTH_TOKEN: Overrides auth_token if set.

Precedence: Constructor arguments take precedence over environment variables. If neither is provided for server_url, a ValueError will be raised.

Error Handling
The library includes robust error handling for network requests. If the logging server is unavailable, times out, or returns an HTTP error (4xx/5xx), a warning or error message will be printed to the console, but your training script will continue to run without interruption.

Example error messages you might see:

Warning: Request to http://localhost:5000/api/logs timed out for step 10. The server did not respond within the expected time.

Error: Could not connect to server at http://localhost:9999/api/logs for step N/A. The server might be unavailable or the URL is incorrect. Error details: ...

Error: HTTP error occurred while sending logs for step 20. Status: 401, Response: Unauthorized. Check server logs for more details.

Local Testing with a Dummy Server
For local development and testing, you can use a simple Flask server to receive and print the logs:

Save dummy_server.py:

# dummy_server.py
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/api/logs', methods=['POST'])
def receive_logs():
    data = request.get_json()
    print("\n--- Received data ---")
    print(json.dumps(data, indent=2))
    # Check for Authorization header
    auth_header = request.headers.get('Authorization')
    if auth_header:
        print(f"Authorization Header: {auth_header}")
    print("---------------------\n")
    return jsonify({"status": "success", "message": "Log received"}), 200

if __name__ == '__main__':
    print("Dummy server running on http://127.0.0.1:5000/api/logs")
    app.run(port=5000)

Install Flask:

pip install Flask

Run the dummy server:

python dummy_server.py

Run your training script (or the example usage from messenger_logger/callback.py's if __name__ == "__main__": block). You will see the logs printed in your dummy server's console.

Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests on the GitHub repository.

License
This project is licensed under the MIT License - see the LICENSE file for details.