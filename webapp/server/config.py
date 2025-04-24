# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Working with configuration."""

# pylint: disable=C0330, g-bad-import-order, g-multiple-import, g-importing-member
import argparse
import os
import json
import google.auth
import smart_open
from logger import logger


class InvalidConfigurationError(Exception):
  """Invalid configuration."""


class ApplicationNotInitializedError(Exception):
  """Application is not initialized."""


class ConfigItemBase:
  """Base class for Config."""

  def __init__(self):
    # copy class attributes (with values) into the instance
    members = [
        attr for attr in dir(self)
        if not attr.startswith('__') and attr != 'update' and attr != 'validate'
    ]
    for attr in members:
      setattr(self, attr, getattr(self, attr))

  def update(self, kw: dict):
    """Update current object with values from json/dict.

    Only known properties (i.e. those that exist in object's class
    as class attributes) are set.

    Args:
      kw: dict with values (usually from json load)
    """
    cls = type(self)
    for k in kw:
      if hasattr(cls, k):
        new_val = kw[k]
        def_val = getattr(cls, k)
        if new_val == '' and def_val != '':
          new_val = def_val
        setattr(self, k, new_val)


class Config(ConfigItemBase):
  """Application configuration."""
  project_id: str = ''
  spreadsheet_id: str = ''
  config_location: str = ''

  def to_dict(self) -> dict:
    """Convert to a dictionary."""
    values = {
        'project_id': self.project_id,
        'spreadsheet_id': self.spreadsheet_id,
        'config_location': self.config_location,
    }
    return values


def parse_arguments(only_known: bool = True) -> argparse.Namespace:
  """Initialize command line parser using argparse.

  Args:
    only_known: true to parse only known properties

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', help='Config file path')
  parser.add_argument('--project_id', '--project-id', help='GCP project id.')

  if only_known:
    args = parser.parse_known_args()[0]
  else:
    args = parser.parse_args()
  args.config = args.config or os.environ.get('CONFIG') or 'config.json'
  return args


def find_project_id(args: argparse.Namespace):
  if getattr(args, 'project_id', ''):
    project_id = getattr(args, 'project_id')
  else:
    _, project_id = google.auth.default()
  return project_id


def get_config_url(args: argparse.Namespace):
  """Return a config file path.

  Args:
    args: cli arguments.

  Raises
    InvalidConfigurationError: config file path contains PROJECT_ID macro
      but we can't detect a GCP project in the environment.
  """
  config_file_name = args.config or os.environ.get('CONFIG') or 'config.json'
  if config_file_name.find('$PROJECT_ID') > -1:
    project_id = find_project_id(args)
    if project_id is None:
      raise InvalidConfigurationError(
          'Config file url contains macro $PROJECT_ID but '
          "project id isn't specified and can't be detected from environment")
    config_file_name = config_file_name.replace('$PROJECT_ID', project_id)
  return config_file_name


def get_config(args: argparse.Namespace | None = None,
               fail_if_not_exists=False) -> Config:
  """Read config file.

  Read and merge settings from a config file, command line args and env vars.

  Args:
    args: CLI arguments.
    fail_if_not_exists: pass true to raise an exception if no config found.

  Returns:
    a config object
  """
  if not args:
    args = parse_arguments()
  config_file_path = get_config_url(args)
  logger.info('Using config file %s', config_file_path)
  try:
    with smart_open.open(config_file_path, 'rb') as f:
      content = f.read()
  except (FileNotFoundError, google.cloud.exceptions.NotFound) as e:
    msg = f'Config file {config_file_path} was not found: {e}'
    logger.error(msg)
    if fail_if_not_exists:
      raise ApplicationNotInitializedError(msg) from e
    else:
      logger.warning('Config file was not found, using a default one')
      config_file_path = ''
      content = '{}'

  cfg_dict: dict = json.loads(content)
  config = Config()
  config.update(cfg_dict)
  config.config_location = config_file_path

  # project id (CLI arg overwrites config)
  if getattr(args, 'project_id', ''):
    config.project_id = getattr(args, 'project_id')
  if not config.project_id:
    config.project_id = find_project_id(args)

  if not config.project_id:
    logger.error('We could not detect GCP project_id')
  else:
    logger.info('Project id: %s', config.project_id)

  logger.debug(config.to_dict())
  return config


def save_config(config: Config, args: argparse.Namespace | None = None):
  """Save the current config into a file.

  Output location is determined by get_config_url.

  Args:
    config: config object.
    args: CLI arguments.
  """
  if not args:
    args = parse_arguments()
  config_file_name = get_config_url(args)
  with smart_open.open(config_file_name, 'w') as f:
    config_dict = config.to_dict()
    # NOTE: we're not saving the following parameters as they can be detected
    # in runtime and it's be more reliable than taking them from config
    del config_dict['project_id']
    del config_dict['config_location']
    f.write(json.dumps(config_dict))
    logger.info('Config saved to %s', config_file_name)
