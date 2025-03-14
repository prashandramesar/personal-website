# wsgi_app_for_fastapi.py with fixed type annotations
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

from fastapi import FastAPI

# Define a type variable for the response items
T = TypeVar("T")


class MountToWsgi:
    """
    Class to mount a FastAPI app to a WSGI application
    """

    def __init__(self, app: FastAPI, root_path: str = ""):
        self.app = app
        self.root_path = root_path.rstrip("/")

    def __call__(self, environ, start_response):
        """
        WSGI callable
        """
        path = environ.get("PATH_INFO", "")
        if path.startswith(self.root_path):
            # Update path to be relative to mount point
            environ["PATH_INFO"] = path[len(self.root_path) :]
            environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + self.root_path

            # Create an ASGI scope
            scope = self._create_scope(environ)

            # Create an event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the ASGI app with the WSGI interface
            asgi_callable = self.app
            return loop.run_until_complete(
                self._run_asgi(asgi_callable, scope, environ, start_response)
            )

        # Not a match for this app, let the Flask app handle it
        return None

    def _create_scope(self, environ) -> Dict[str, Any]:
        """
        Convert WSGI environ to ASGI scope
        """
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": environ.get("SERVER_PROTOCOL", "HTTP/1.1").split("/")[1],
            "method": environ["REQUEST_METHOD"],
            "scheme": environ.get("wsgi.url_scheme", "http"),
            "path": environ["PATH_INFO"],
            "raw_path": environ["PATH_INFO"].encode("utf-8"),
            "query_string": environ.get("QUERY_STRING", "").encode("utf-8"),
            "root_path": environ.get("SCRIPT_NAME", ""),
            "headers": [
                (
                    key.lower().replace("http_", "").replace("_", "-").encode("utf-8"),
                    value.encode("utf-8"),
                )
                for key, value in environ.items()
                if key.startswith("HTTP_") or key in ("CONTENT_TYPE", "CONTENT_LENGTH")
            ],
            "server": (
                environ.get("SERVER_NAME", ""),
                int(environ.get("SERVER_PORT", 0)),
            ),
            "client": (
                environ.get("REMOTE_ADDR", ""),
                int(environ.get("REMOTE_PORT", 0)),
            ),
        }
        return scope

    async def _run_asgi(
        self,
        app: Callable,
        scope: Dict[str, Any],
        environ: Dict[str, Any],
        start_response: Callable,
    ):
        """
        Run an ASGI application with WSGI interface
        """
        # Create response containers with explicit typing
        status_code: List[Optional[str]] = [None]
        response_headers: List[Optional[List[Tuple[str, str]]]] = [None]
        response_body: List[bytes] = []

        # ASGI receive function
        async def receive() -> Dict[str, Any]:
            body_chunk = environ["wsgi.input"].read(
                int(environ.get("CONTENT_LENGTH", 0) or 0)
            )
            return {
                "type": "http.request",
                "body": body_chunk,
                "more_body": False,
            }

        # ASGI send function
        async def send(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_code[0] = str(message["status"])
                response_headers[0] = [
                    (key.decode("utf-8"), value.decode("utf-8"))
                    for key, value in message.get("headers", [])
                ]
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    response_body.append(body)

        # Run the ASGI app
        await app(scope, receive, send)

        # Convert to WSGI response
        if status_code[0] is not None and response_headers[0] is not None:
            start_response(
                f"{status_code[0]} {self._status_phrases.get(int(status_code[0]), 'Unknown')}",
                cast(List[Tuple[str, str]], response_headers[0]),
            )
            return [b"".join(response_body)]

        # Default empty response
        start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
        return [b"ASGI application did not complete"]

    # Common HTTP status phrases
    _status_phrases = {
        100: "Continue",
        101: "Switching Protocols",
        102: "Processing",
        200: "OK",
        201: "Created",
        202: "Accepted",
        203: "Non-Authoritative Information",
        204: "No Content",
        205: "Reset Content",
        206: "Partial Content",
        207: "Multi-Status",
        208: "Already Reported",
        226: "IM Used",
        300: "Multiple Choices",
        301: "Moved Permanently",
        302: "Found",
        303: "See Other",
        304: "Not Modified",
        305: "Use Proxy",
        307: "Temporary Redirect",
        308: "Permanent Redirect",
        400: "Bad Request",
        401: "Unauthorized",
        402: "Payment Required",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        406: "Not Acceptable",
        407: "Proxy Authentication Required",
        408: "Request Timeout",
        409: "Conflict",
        410: "Gone",
        411: "Length Required",
        412: "Precondition Failed",
        413: "Payload Too Large",
        414: "URI Too Long",
        415: "Unsupported Media Type",
        416: "Range Not Satisfiable",
        417: "Expectation Failed",
        418: "I'm a teapot",
        421: "Misdirected Request",
        422: "Unprocessable Entity",
        423: "Locked",
        424: "Failed Dependency",
        425: "Too Early",
        426: "Upgrade Required",
        428: "Precondition Required",
        429: "Too Many Requests",
        431: "Request Header Fields Too Large",
        451: "Unavailable For Legal Reasons",
        500: "Internal Server Error",
        501: "Not Implemented",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
        505: "HTTP Version Not Supported",
        506: "Variant Also Negotiates",
        507: "Insufficient Storage",
        508: "Loop Detected",
        510: "Not Extended",
        511: "Network Authentication Required",
    }
