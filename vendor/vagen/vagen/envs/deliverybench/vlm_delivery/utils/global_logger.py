# utils/global_logger.py
# -*- coding: utf-8 -*-

"""
Global logging module.

Provides a unified logging interface across the entire project.
Includes a singleton logger, configuration utilities, and
convenience helpers for subsystem- and agent-specific loggers.
"""

import logging
import os
from datetime import datetime
from typing import Optional


class GlobalLogger:
    """
    Singleton provider for a project-wide logger.

    This class initializes a root logger that writes to:
      - A timestamped log file.
      - The console (stdout).

    It supports dynamic reconfiguration and guarantees
    that setup occurs only once.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Ensure a single shared instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the logger on first access."""
        if not GlobalLogger._initialized:
            self._setup_logger()
            GlobalLogger._initialized = True

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _setup_logger(
        self,
        log_folder: str = "../../log",
        file_log_level: int = logging.DEBUG,
        console_log_level: int = logging.INFO,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        """
        Configure the global logger.

        Args:
            log_folder: Directory where log files are stored.
            file_log_level: Logging level for the log file.
            console_log_level: Logging level for console output.
            log_format: Format string for log messages.
            date_format: Timestamp format for logged entries.
        """
        # Ensure log directory exists
        os.makedirs(log_folder, exist_ok=True)

        # Timestamped log filename
        log_file = os.path.join(
            log_folder, f"delivery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Create root logger
        self.logger = logging.getLogger("delivery_system")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)

        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Reduce verbosity in noisy external modules
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a namespaced child logger.

        Args:
            name: Logger namespace string.

        Returns:
            A child `logging.Logger` instance.
        """
        return self.logger.getChild(name)

    def configure(
        self,
        log_folder: Optional[str] = None,
        file_log_level: Optional[int] = None,
        console_log_level: Optional[int] = None,
        log_format: Optional[str] = None,
        date_format: Optional[str] = None,
    ):
        """
        Reconfigure the global logger.

        Any omitted argument falls back to the default configuration.

        Args:
            log_folder: Directory for log outputs.
            file_log_level: File logging level.
            console_log_level: Console logging level.
            log_format: Log message format string.
            date_format: Timestamp display format.
        """
        if GlobalLogger._initialized:
            self._setup_logger(
                log_folder=log_folder or "../../log",
                file_log_level=file_log_level or logging.DEBUG,
                console_log_level=console_log_level or logging.INFO,
                log_format=log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                date_format=date_format or "%Y-%m-%d %H:%M:%S",
            )


# ----------------------------------------------------------------------
# Global instance and convenience wrappers
# ----------------------------------------------------------------------

_global_logger = GlobalLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to retrieve a child logger.

    Args:
        name: Namespace for the logger.

    Returns:
        A configured logger instance.
    """
    return _global_logger.get_logger(name)


def configure_logger(**kwargs):
    """
    Convenience wrapper to reconfigure the global logger.

    Args:
        **kwargs: Forwarded to GlobalLogger.configure().
    """
    _global_logger.configure(**kwargs)


# ----------------------------------------------------------------------
# Predefined logger helpers
# ----------------------------------------------------------------------

def get_agent_logger(agent_id: str) -> logging.Logger:
    """Return a logger namespaced for a specific agent."""
    return get_logger(f"agent_{agent_id}")


def get_system_logger() -> logging.Logger:
    """Return the system-level logger."""
    return get_logger("system")


def get_vlm_logger() -> logging.Logger:
    """Return a logger used for VLM-related operations."""
    return get_logger("vlm")