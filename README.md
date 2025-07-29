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
