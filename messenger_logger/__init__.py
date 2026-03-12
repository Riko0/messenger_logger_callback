from .logger import MessengerLogger

try:
    from .callback import MessengerLoggerCallback
except ImportError:
    pass
