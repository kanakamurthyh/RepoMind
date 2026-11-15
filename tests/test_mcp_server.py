from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")


def test_mcp_server_exposes_repository_tools(shop_env):
    from app.api.services import ServiceContainer
    from app.mcp_server import build_server

    analyzer, chat, repo_id = shop_env
    services = ServiceContainer(analyzer.settings)
    services.analyzer = analyzer
    services.chat = chat
    server = build_server(services)

    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {"analyze_repository", "search_code", "ask_repository", "investigate", "find_symbol", "read_file", "repository_insights"} <= names
