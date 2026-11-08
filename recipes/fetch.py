"""Fetch a recipe page from a URL someone typed, without trusting that URL.

Every hop is checked before connecting, and the connection goes to the IP that
was checked, so a hostname can't resolve to a public address during the check
and a private one for the request. Page JavaScript never runs.
"""

import ipaddress
import os
import socket
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

MAX_PAGE_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
TOTAL_TIMEOUT_SECONDS = 20.0
ALLOWED_PORTS = {80, 443}
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
# How a site says it won't serve automated requests. These are never retried or
# worked around.
BLOCKED_STATUSES = {401, 403, 407, 429, 451, 503}
HTML_TYPES = ("text/html", "application/xhtml+xml")
OWN_DOMAIN = "prashand.nl"
USER_AGENT = (
    "Mozilla/5.0 (compatible; prashand-recipe-tool/1.0; +https://www.prashand.nl)"
)
TRACKING_PARAMS = {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "ref"}
# Their terms bar scraping, so these are never requested, even after a redirect
# from a short link. Their posts are pasted instead.
SOCIAL_DOMAINS = {
    "facebook.com": "Facebook",
    "fb.watch": "Facebook",
    "instagram.com": "Instagram",
    "threads.net": "Threads",
    "tiktok.com": "TikTok",
    "twitter.com": "X",
    "x.com": "X",
    "youtu.be": "YouTube",
    "youtube.com": "YouTube",
}


class FetchError(Exception):
    """A page that couldn't be fetched, with a message that's safe to show."""


class UnsafeUrl(FetchError):
    """A URL the tool refuses to request."""


class BlockedBySite(FetchError):
    """The site refused the request, so the text has to be pasted instead."""


@dataclass(frozen=True)
class FetchedPage:
    url: str
    html: str


def canonicalize_url(url: str) -> str:
    """Normalize a URL so that the same recipe is only imported and paid for once."""
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS and not key.lower().startswith("utm_")
    ]
    path = parts.path.rstrip("/")
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), "")
    )


def _is_own_host(host: str) -> bool:
    """Never request this site itself. That caused an infinite loop before."""
    own_hosts = {"localhost", OWN_DOMAIN}
    render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
    if render_host:
        own_hosts.add(render_host.lower())
    return any(host == own or host.endswith(f".{own}") for own in own_hosts)


def _checked_ip(host: str, port: int) -> str:
    """Resolve a host and return its IP if every address it has is public."""
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as e:
        raise FetchError("That site couldn't be found.") from e
    ips = [
        ipaddress.ip_address(str(address[4][0]).split("%")[0]) for address in addresses
    ]
    if not ips:
        raise FetchError("That site couldn't be found.")
    for ip in ips:
        mapped = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) else None
        checked = mapped or ip
        if not checked.is_global or checked.is_multicast:
            raise UnsafeUrl("That address isn't a public website.")
    return str(ips[0])


def _checked_target(url: str) -> tuple[httpx.URL, str]:
    """Validate a URL and return it with the IP to connect to."""
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, TypeError) as e:
        raise UnsafeUrl("That isn't a valid link.") from e
    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise UnsafeUrl("Only http and https links can be imported.")
    if parsed.username or parsed.password:
        raise UnsafeUrl("Links containing a username or password aren't allowed.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise UnsafeUrl("Only links on the standard web ports can be imported.")
    host = parsed.host.lower()
    if _is_own_host(host):
        raise UnsafeUrl("This site can't import its own pages.")
    for domain, platform in SOCIAL_DOMAINS.items():
        if host == domain or host.endswith(f".{domain}"):
            raise BlockedBySite(f"{platform} posts can't be read automatically.")
    return parsed, _checked_ip(host, port)


def _read_limited(response: httpx.Response, deadline: float) -> bytes:
    """Read a body, stopping at the size cap or the deadline."""
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_PAGE_BYTES:
        raise FetchError("That page is too large to import.")
    chunks: list[bytes] = []
    total = 0
    # iter_bytes decompresses, so this also limits compressed bombs.
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > MAX_PAGE_BYTES:
            raise FetchError("That page is too large to import.")
        if time.monotonic() > deadline:
            raise FetchError("That site took too long to respond.")
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_page(url: str, transport: httpx.BaseTransport | None = None) -> FetchedPage:
    """Fetch an HTML page. Raises FetchError with a message that's safe to show."""
    deadline = time.monotonic() + TOTAL_TIMEOUT_SECONDS
    current = url.strip()
    with httpx.Client(
        transport=transport,
        follow_redirects=False,
        timeout=httpx.Timeout(10.0),
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            parsed, ip = _checked_target(current)
            host = parsed.host
            # Connect to the checked IP, while TLS still verifies the real name.
            pinned = parsed.copy_with(host=ip)
            netloc = parsed.netloc.decode("ascii")
            try:
                with client.stream(
                    "GET",
                    pinned,
                    headers={"Host": netloc},
                    extensions={"sni_hostname": host},
                ) as response:
                    if response.status_code in REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError("That site sent a broken redirect.")
                        current = urljoin(current, location)
                        continue
                    if response.status_code in BLOCKED_STATUSES:
                        raise BlockedBySite(
                            "This site doesn't allow automated requests "
                            f"(status {response.status_code})."
                        )
                    if response.status_code != 200:
                        raise FetchError(
                            f"That page returned status {response.status_code}."
                        )
                    content_type = response.headers.get("content-type", "").lower()
                    if not content_type.startswith(HTML_TYPES):
                        raise FetchError("That link isn't a web page.")
                    body = _read_limited(response, deadline)
                    encoding = response.encoding or "utf-8"
            except httpx.TimeoutException as e:
                raise FetchError("That site took too long to respond.") from e
            except httpx.HTTPError as e:
                raise FetchError("That site couldn't be reached.") from e
            try:
                html = body.decode(encoding, errors="replace")
            except LookupError:
                html = body.decode("utf-8", errors="replace")
            return FetchedPage(current, html)
    raise FetchError("That link redirects too many times.")
