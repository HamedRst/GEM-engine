from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "process": record.process,
            "process_name": record.processName,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(
    output_dir: str | Path,
    level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
    structured_jsonl: bool = True,
    max_file_mb: int = 25,
    backup_count: int = 5,
) -> logging.Logger:
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gem_learning_engine")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    text_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if log_to_console:
        console = logging.StreamHandler()
        console.setFormatter(text_formatter)
        logger.addHandler(console)
    if log_to_file:
        file_handler = RotatingFileHandler(
            log_dir / "pipeline.log",
            maxBytes=max_file_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(text_formatter)
        logger.addHandler(file_handler)
    if structured_jsonl:
        json_handler = RotatingFileHandler(
            log_dir / "pipeline.jsonl",
            maxBytes=max_file_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        json_handler.setFormatter(JsonLineFormatter())
        logger.addHandler(json_handler)
    return logger
