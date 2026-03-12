# Messenger Logger Callback

A Python library for sending training logs to a remote server. Works as a **standalone logger** for any training loop or as a **Hugging Face Trainer Callback**.

## Installation

```bash
# Standalone (no transformers dependency)
pip install messenger-logger-callback

# With Hugging Face Trainer support
pip install messenger-logger-callback[trainer]
```

## Quick Start: Standalone Logger

Use `MessengerLogger` in any training loop — plain PyTorch, Lightning, or anything else.

```python
from messenger_logger import MessengerLogger

logger = MessengerLogger(
    server_url="http://your-server:5000/api/logs",
    project_name="resnet_experiment",
    run_id="run_v3",
    author_username="riko",
)

logger.start()

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)
        global_step += 1
        logger.log(step=global_step, epoch=epoch, metrics={"loss": loss.item()})
    logger.epoch_end(epoch)

logger.finish()
```

### Available Methods

| Method | Description |
|--------|-------------|
| `start()` | Signal training start. Begins heartbeat. |
| `log(step, metrics, epoch=None)` | Log training metrics at a given step. |
| `epoch_end(epoch)` | Signal end of an epoch. |
| `log_custom(data)` | Send arbitrary custom data. |
| `finish()` | Signal training end. Stops heartbeat. |

## Quick Start: Hugging Face Trainer

Use `MessengerLoggerCallback` as a drop-in Trainer callback.

```python
from transformers import Trainer, TrainingArguments
from messenger_logger.callback import MessengerLoggerCallback

logger = MessengerLoggerCallback(
    server_url="http://your-server:5000/api/logs",
    project_name="bert_finetune",
    run_id="experiment_v2",
    auth_token="your_secret_token",
    author_username="riko",
    metadata={"model": "bert-large", "dataset": "squad"},
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[logger],
)
trainer.train()
```

The callback automatically captures `on_log`, `on_train_begin`, `on_train_end`, and `on_epoch_end` events.

You can also send custom data at any point:

```python
logger.send_custom_log({
    "event": "evaluation_complete",
    "accuracy": 0.95,
    "f1": 0.93,
})
```

## Heartbeat

A background thread sends a lightweight heartbeat signal to the server every 60 seconds by default. This allows the server to detect crashed or stalled runs much faster than waiting for missing log events.

- **On by default** with a 60-second interval.
- Starts when training begins, stops when training ends.
- If the training process crashes, the heartbeat stops automatically.

To change the interval or disable:

```python
# Custom interval (30 seconds)
logger = MessengerLogger(server_url="...", heartbeat_interval=30)

# Disable heartbeat
logger = MessengerLogger(server_url="...", heartbeat_interval=None)
```

## ClearML Integration

If [ClearML](https://clear.ml/) is active in your training script, the library automatically detects the current task and includes a link to the ClearML dashboard in every payload sent to the server. No configuration needed — if `clearml` is installed and a task is running, the link is captured.

## Configuration

Settings can be provided via constructor arguments, environment variables, or a `.env` file. Constructor arguments take highest precedence, then environment variables, then `.env` file values.

| Constructor Argument | Environment Variable | Description |
|---------------------|---------------------|-------------|
| `server_url` | `MESSENGER_LOGGER_SERVER_URL` | HTTP endpoint to send logs to. **Required.** |
| `project_name` | — | Project identifier. Default: `"default_project"`. |
| `run_id` | — | Unique run identifier. Auto-generated if omitted. |
| `auth_token` | `MESSENGER_LOGGER_AUTH_TOKEN` | Bearer token for the Authorization header. |
| `author_username` | `MESSENGER_LOGGER_AUTHOR_USERNAME` | Who started the run. Default: `"anonymous"`. |
| `metadata` | `MESSENGER_LOGGER_METADATA` | Static metadata dict. Env var should be a JSON string. |
| `dotenv_path` | `MESSENGER_LOGGER_DOTENV` | Path to a `.env` file to load config from. |
| `heartbeat_interval` | — | Seconds between heartbeats. Default: `60`. Set to `None` to disable. |

## Error Handling

Network errors (timeouts, connection failures, HTTP errors) are caught and printed as warnings. They never crash your training. Example messages:

```
Warning: Request to http://... timed out for step 10.
Error: Could not connect to server at http://... for step 20.
Error: HTTP error occurred while sending logs for step 30. Status: 401.
```

## License

MIT
