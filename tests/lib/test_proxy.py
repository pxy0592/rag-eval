import os

from src.lib.proxy import remove_unsupported_socks_proxies


def test_remove_legacy_socks_proxy(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7892")
    monkeypatch.setenv("all_proxy", "socks://127.0.0.1:7892")

    remove_unsupported_socks_proxies()

    assert "ALL_PROXY" not in os.environ
    assert "all_proxy" not in os.environ


def test_keep_supported_proxy_values(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:7892")
    monkeypatch.setenv("all_proxy", "socks5://127.0.0.1:7892")

    remove_unsupported_socks_proxies()

    assert os.environ["ALL_PROXY"] == "http://127.0.0.1:7892"
    assert os.environ["all_proxy"] == "socks5://127.0.0.1:7892"
