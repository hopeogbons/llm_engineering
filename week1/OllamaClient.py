import os
import requests
import urllib.parse
from typing import Literal, List, Dict, Optional
import ollama

class OllamaClient:
    """
    Robust wrapper for Ollama that:
      - normalizes host (removes trailing /v1),
      - performs a small health-check,
      - injects Authorization header safely for cloud mode,
      - gives clear errors for common connectivity problems.

      Created by Hope Ogbons
      Date: 2025-10-19
      Version: 1.0.0
      License: MIT
      Copyright (c) 2025 Hope Ogbons
      Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
      The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """
    DEFAULT_LOCAL = "http://127.0.0.1:11434"
    DEFAULT_CLOUD = "https://api.ollama.com"

    # Note: Both "cloud" and "local" modes are accessed through "local". This is just a future proofing measure.
    def __init__(self, api_key: Optional[str] = None, mode: Literal["local","cloud"]="local", host: Optional[str]=None, health_timeout: float = 2.0) -> None:
        self.mode = mode.lower()
        if self.mode not in {"local","cloud"}:
            raise ValueError("mode must be 'local' or 'cloud'")

        # allow overriding host (useful for docker/host.docker.internal, WSL, etc)
        if host:
            self.host = self._normalize_host(host)
        else:
            self.host = self._normalize_host(self.DEFAULT_CLOUD if self.mode=="cloud" else self.DEFAULT_LOCAL)

        self.health_timeout = health_timeout
        self._requests_session = requests.Session()

        # create SDK client using normalized host (no trailing /v1)
        self.client = ollama.Client(host=self.host)

        # set Authorization header if cloud
        if self.mode == "cloud":
            token = api_key or os.environ.get("OLLAMA_API_KEY")
            if not token:
                raise ValueError("api_key is required for cloud mode (or set OLLAMA_API_KEY)")
            self._set_auth_header(token)

    def _normalize_host(self, host: str) -> str:
        """Remove trailing /v1 or trailing slash so SDK won't double-up paths."""
        # ensure scheme present
        parsed = urllib.parse.urlparse(host if "://" in host else "https://" + host)
        base = f"{parsed.scheme}://{parsed.netloc}"
        # if path contains something like /v1, ignore it
        return base

    def _set_auth_header(self, token: str) -> None:
        """Try some common places the SDK stores a requests-like session/headers."""
        header_value = f"Bearer {token}"
        # 1) try common internal attribute name used by some SDKs
        try:
            if hasattr(self.client, "_client") and hasattr(self.client._client, "headers"):
                self.client._client.headers.update({"Authorization": header_value})
                return
        except Exception:
            pass
        # 2) try if the client itself exposes headers
        try:
            if hasattr(self.client, "headers"):
                self.client.headers.update({"Authorization": header_value})
                return
        except Exception:
            pass
        # 3) fall back to requests session for health checks and raise a friendly warning
        self._requests_session.headers.update({"Authorization": header_value})

    def healthcheck(self) -> Dict[str, object]:
        """
        Performs a lightweight GET against <host>/v1/models to verify the server is reachable.
        Returns a dict with status and details.
        """
        url = f"{self.host}/v1/models"
        try:
            resp = self._requests_session.get(url, timeout=self.health_timeout)
            return {"ok": resp.ok, "status_code": resp.status_code, "url": url, "text_snippet": resp.text[:300]}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "error": str(e), "url": url}

    def chat(self, model: str, messages: List[Dict[str,str]], stream: bool=False) -> dict:
        """
        Wraps the SDK call and surfaces clearer errors for connection problems.
        """
        # quick pre-flight
        hc = self.healthcheck()
        if not hc.get("ok"):
            # include helpful hints in the exception
            hint = (
                f"Healthcheck failed for {hc.get('url')!s}: {hc.get('error') or hc.get('status_code')}. "
                "If running locally, ensure `ollama serve` is running on that host and port. "
                "If inside Docker/WSL, ensure the host is reachable from the container (use host.docker.internal or expose 0.0.0.0)."
            )
            raise ConnectionError(hint)

        try:
            return self.client.chat(model=model, messages=messages, stream=stream)
        except Exception as e:
            # Commonly this will be an underlying requests/httpx ConnectionError
            raise ConnectionError(
                "Failed to call Ollama API. Common causes: server not running, wrong host/port, firewall, or SDK-client mismatch. "
                f"Underlying error: {e}"
            ) from e
