# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Application logger setup."""

# pylint: disable=C0330, g-bad-import-order, g-multiple-import, g-importing-member
import os
import logging
import env
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

LOGGER_NAME = 'GeoFlex'

logging.basicConfig(
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

loglevel = os.getenv('LOG_LEVEL') or 'DEBUG'
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(loglevel)

if env.IS_GAE:
  import google.cloud.logging
  from google.cloud.logging.handlers import CloudLoggingHandler, setup_logging

  def _converter(o):
    if isinstance(o, datetime):
      return o.isoformat()
    if isinstance(o, set):
      return list(o)
    if isinstance(o, dict):
      return {_converter(k): _converter(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
      return [_converter(i) for i in o]
    return o

  class SetToListFilter(logging.Filter):
    def filter(self, record):
      if isinstance(record.msg, (dict, list, tuple, set)):
        record.msg = _converter(record.msg)
      if record.args:
        record.args = tuple(_converter(list(record.args)))
      return True

  # Prevent duplicate handlers in Gunicorn worker processes
  if not any(
      isinstance(h, CloudLoggingHandler) for h in logging.getLogger().handlers):
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client, name=LOGGER_NAME)
    handler.addFilter(SetToListFilter())
    handler.setLevel(loglevel)
    setup_logging(handler)


@dataclass
class LogEntry:
  """Represents a single log entry with all relevant information."""
  timestamp: datetime
  level: str
  logger_name: str
  message: str
  module: str | None = None
  function: str | None = None
  line_number: str | None = None

  def to_dict(self) -> dict:
    """Convert log entry to dictionary for JSON serialization."""
    return {
        'timestamp': self.timestamp.isoformat(),
        'level': self.level,
        'logger_name': self.logger_name,
        'message': self.message,
        'module': self.module,
        'function': self.function,
        'line_number': self.line_number
    }


def logs_to_dict_list(logs: list[LogEntry]) -> list[dict]:
  """Convert list of LogEntry objects to list of dictionaries for JSON response."""
  return [log_entry.to_dict() for log_entry in logs]


class LogInterceptor(logging.Handler):
  """Custom logging handler that captures log records matching a prefix."""

  def __init__(self, prefix: str, log_list: list[LogEntry]):
    super().__init__()
    self.prefix = prefix
    self.log_list = log_list

  def emit(self, record: logging.LogRecord):
    """Handle a log record if it matches the prefix."""
    if record.name.startswith(self.prefix):
      try:
        # Create log entry with detailed information
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc),
            level=record.levelname,
            logger_name=record.name,
            message=self.format(record),
            module=getattr(record, 'module', None),
            function=getattr(record, 'funcName', None),
            line_number=getattr(record, 'lineno', None))

        # Append to log list
        self.log_list.append(log_entry)

      except Exception as e:
        # Prevent logging errors from breaking the interceptor
        print(f"Error in log interceptor: {e}", file=sys.stderr)


@contextmanager
def intercept_logs(prefix: str, logs: list[LogEntry], level: int | None = None):
  """
    Context manager to intercept logs from loggers with specified prefix.

    Args:
        prefix: Logger name prefix to match (e.g., 'geoflex.')
        logs: List to append captured log entries to
        level: Minimum logging level to capture (default: DEBUG)
    """

  # Create the interceptor handler
  interceptor = LogInterceptor(prefix, logs)
  if level:
    interceptor.setLevel(level)

  root_logger = logging.getLogger()
  original_level = root_logger.level

  try:
    # Set root logger to capture all levels if needed
    if level and root_logger.level > level:
      root_logger.setLevel(level)

    root_logger.addHandler(interceptor)

    yield

  finally:
    # Clean up: remove our handler
    root_logger.removeHandler(interceptor)

    # Restore original logging level
    if level:
      root_logger.setLevel(original_level)
