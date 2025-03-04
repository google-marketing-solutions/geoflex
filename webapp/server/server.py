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

# pylint: disable=C0330, g-bad-import-order, g-multiple-import
from typing import Any
import json
import os
import math
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from env import IS_GAE
from logger import logger

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class CustomJSONEncoder(json.JSONEncoder):
  """A custom JSON encoder to support serialization of special objects."""

  def default(self, o: Any):
    # Handle regular Python floats
    if isinstance(o, float):
      if math.isinf(o):
        return "Infinity" if o > 0 else "-Infinity"
      if math.isnan(o):
        return "NaN"

      # For other types, use default serialization
      return super().default(o)


class CustomJSONResponse(JSONResponse):
  """Custom JSON response using our JSON encoder."""

  def render(self, content: Any) -> bytes:
    return json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=True,
        indent=None,
        separators=(",", ":"),
        cls=CustomJSONEncoder,
    ).encode("utf-8")


STATIC_DIR = (os.getenv("STATIC_DIR") or "../dist"
             )  # folder for static content relative to the current module
# Calculate the absolute path based on the current file location
STATIC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), STATIC_DIR))


def safe_remove_header(headers, header_name):
  if header_name in headers:
    headers.remove(header_name)


@app.get("/{full_path:path}")
async def catch_all(full_path: str) -> Response:
  """Serve static files or SPA index.html."""
  # Handle API paths
  if full_path.startswith("api/"):
    raise StarletteHTTPException(
        status_code=404, detail="API endpoint not found")

  # Check if the requested file exists
  file_path = os.path.join(STATIC_PATH, full_path)

  if os.path.isfile(file_path):
    # Serve the file directly
    response = FileResponse(file_path)

    # Handle GAE timestamp issues
    if IS_GAE:
      safe_remove_header(response.headers, "last-modified")

    return response

  # If file not found, serve index.html (SPA behavior)
  index_path = os.path.join(STATIC_PATH, "index.html")

  if not os.path.exists(index_path):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Frontend application files not found: " + index_path
        })

  response = FileResponse(index_path)

  safe_remove_header(response.headers, "etag")
  response.headers["Cache-Control"] = "no-cache, no-store"

  if IS_GAE:
    safe_remove_header(response.headers, "last-modified")

  return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request,
                                       exc: RequestValidationError) -> Response:
  """Handle validation errors."""
  logger.error("Validation error: %s", exc)
  return JSONResponse(
      status_code=422,
      content={"error": {
          "type": "ValidationError",
          "message": str(exc)
      }},
  )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_: Request,
                                 exc: StarletteHTTPException) -> Response:
  """Handle HTTP exceptions."""
  logger.error("HTTP exception: %s", exc.detail)
  return JSONResponse(
      status_code=exc.status_code,
      content={"error": {
          "type": "HTTPException",
          "message": exc.detail
      }},
  )


@app.exception_handler(Exception)
async def general_exception_handler(_: Request, exc: Exception) -> Response:
  """Handle all other exceptions."""
  logger.exception(exc)

  error_type = type(exc).__name__
  error_message = str(exc)

  is_debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "on")

  return JSONResponse(
      status_code=500,
      content={
          "error": {
              "type":
                  error_type,
              "message":
                  f"{error_type}: {error_message}",
              "debugInfo":
                  "".join(traceback.format_tb(exc.__traceback__))
                  if is_debug else "Enable DEBUG mode to see traceback",
          }
      })


if __name__ == "__main__":
  # Run the server when executed directly
  uvicorn.run(
      "server:app",
      host="127.0.0.1",
      port=8080,
      reload=os.getenv("DEBUG", "False").lower() in ("true", "1", "on"))
