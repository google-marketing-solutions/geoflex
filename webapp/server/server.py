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
"""Application server."""

from typing import Any, Callable
import json
import os
import math
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask.json.provider import DefaultJSONProvider
from google.appengine.api import wrap_wsgi_app
from flask_cors import CORS

from env import IS_GAE
from logger import logger

# make linter happy (avoid import-member)
# date = datetime.date
# timedelta = datetime.timedelta
# datetime = datetime.datetime


class JsonEncoder(json.JSONEncoder):
  """A custom JSON encoder to support serialization of Audience objects."""
  flask_default: Callable[[Any], Any]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.flask_default = DefaultJSONProvider.default

  def default(self, o):
    # Handle numpy types first
    # if isinstance(o, np.floating):
    #   if np.isfinite(o):
    #     return float(o)
    #   if np.isinf(o):
    #     return 'Infinity' if o > 0 else '-Infinity'
    #   if np.isnan(o):
    #     return 'NaN'

    # Handle regular Python floats
    if isinstance(o, float):
      if math.isinf(o):
        return 'Infinity' if o > 0 else '-Infinity'
      if math.isnan(o):
        return 'NaN'

    # Handle numpy arrays
    # if isinstance(o, np.ndarray):
    #   return o.tolist()

    # if isinstance(o, (models.FeatureMetrics, models.DistributionData)):
    #   # Convert to dict and recursively handle numpy values
    #   return {
    #       k: self.default(v) if isinstance(v, (np.floating, np.ndarray)) else v
    #       for k, v in o.__dict__.items()
    #   }

    # if isinstance(o, models.Audience):
    #   return o.to_dict()
    return self.flask_default(o)


class JSONProvider(DefaultJSONProvider):
  """A JSON provider to replace JsonEncoder used by Flask."""

  def dumps(self, obj: Any, **kwargs: Any) -> str:
    """Serialize data as JSON to a string.

    Keyword arguments are passed to :func:`json.dumps`. Sets some
    parameter defaults from the :attr:`default`,
    :attr:`ensure_ascii`, and :attr:`sort_keys` attributes.

    :param obj: The data to serialize.
    :param kwargs: Passed to :func:`json.dumps`.
    """
    kwargs.setdefault('cls', JsonEncoder)
    kwargs.setdefault('default', None)
    return DefaultJSONProvider.dumps(self, obj, **kwargs)


STATIC_DIR = (os.getenv('STATIC_DIR') or '../dist'
             )  # folder for static content relative to the current module

Flask.json_provider_class = JSONProvider
app = Flask(__name__)
app.wsgi_app = wrap_wsgi_app(app.wsgi_app)

CORS(app)

#args = parse_arguments(only_known=True)


def _get_req_arg_str(name: str):
  arg = request.args.get(name)
  if not arg or arg == 'null' or arg == 'undefined':
    return None
  return str(arg)


@app.route('/api/datasources', methods=['GET'])
def get_configuration():

  return jsonify({})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
  # NOTE: we don't use Flask standard support for static files
  # (static_folder option and send_static_file method)
  # because they can't distinguish requests for static files (js/css)
  # and client routes (like /products)
  file_requested = os.path.join(app.root_path, STATIC_DIR, path)
  if not os.path.isfile(file_requested):
    path = 'index.html'
  max_age = 0 if path == 'index.html' else None
  response = send_from_directory(STATIC_DIR, path, max_age=max_age)
  # There is a "feature" in GAE - all files have zeroed timestamp
  # ("Tue, 01 Jan 1980 00:00:01 GMT")
  if IS_GAE:
    response.headers.remove('Last-Modified')
  if path == 'index.html':
    response.headers.remove('ETag')
  response.cache_control.no_cache = True
  response.cache_control.no_store = True
  logger.debug('Static file request %s processed', path)
  return response


@app.errorhandler(Exception)
def handle_exception(e: Exception):
  logger.exception(e)
  if getattr(e, 'errors', None):
    logger.error(e.errors)
  if request.accept_mimetypes.accept_json and request.path.startswith('/api/'):
    # NOTE: not all exceptions can be serialized
    error_type = type(e).__name__
    error_message = str(e)

    # format the error message with the traceback
    debug_info = ''
    if app.config['DEBUG']:
      debug_info = ''.join(traceback.format_tb(e.__traceback__))

    # create and return the JSON response
    response = jsonify({
        'error': {
            'type': error_type,
            'message': f'{error_type}: {error_message}',
            'debugInfo': debug_info,
        }
    })
    response.status_code = 500
    return response
    # try:
    #   return jsonify({"error": e}), 500
    # except:
    #   return jsonify({"error": str(e)}), 500
  return e


if __name__ == '__main__':
  # This is used when running locally only. When deploying to Google App
  # Engine, a webserver process such as Gunicorn will serve the app. This
  # can be configured by adding an `entrypoint` to app.yaml.
  app.run(host='127.0.0.1', port=8080, debug=True)
