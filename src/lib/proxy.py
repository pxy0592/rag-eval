"""Compatibility helpers for proxy environment variables used by HTTP clients."""

import os


_PROXY_ENV_NAMES = ("ALL_PROXY", "all_proxy")


def remove_unsupported_socks_proxies() -> None:
    """Prevent HTTPX imports from failing on legacy ``socks://`` proxies.

    HTTPX expects HTTP(S) proxies or SOCKS URLs with the optional SOCKS
    transport installed. The common ``socks://`` value is not a valid HTTPX
    proxy scheme, so remove it and let endpoint-specific HTTP(S) proxy
    variables (if present) take effect.
    """

    for name in _PROXY_ENV_NAMES:
        value = os.environ.get(name, "")
        if value.lower().startswith("socks://"):
            os.environ.pop(name, None)
