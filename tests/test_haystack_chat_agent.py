"""Tests for Haystack Chat Agent integration."""

from unittest.mock import MagicMock, patch
from haystack.dataclasses import ChatMessage
from src.haystack_chat_agent import (
    TOOLS,
    build_agent,
    get_current_time,
    add_numbers,
    search_knowledge_base,
    list_knowledge_base_documents,
    get_document_details,
)


def test_tools_registered():
    tool_names = [t.name for t in TOOLS]
    assert "search_knowledge_base" in tool_names
    assert "list_knowledge_base_documents" in tool_names
    assert "get_document_details" in tool_names
    assert "get_current_time" in tool_names
    assert "add_numbers" in tool_names


@patch("src.haystack_chat_agent._rag_service")
def test_search_knowledge_base_tool(mock_rag):
    mock_rag.search_and_format.return_value = "Mocked KB Results"
    result = search_knowledge_base.function(query="architecture guidelines", limit=5)
    assert result == "Mocked KB Results"
    mock_rag.search_and_format.assert_called_once_with(query="architecture guidelines", limit=5)


@patch("src.haystack_chat_agent._rag_service")
def test_list_knowledge_base_documents_tool(mock_rag):
    mock_rag.list_documents_formatted.return_value = "Indexed Docs: Doc1"
    result = list_knowledge_base_documents.function(limit=20)
    assert result == "Indexed Docs: Doc1"
    mock_rag.list_documents_formatted.assert_called_once_with(limit=20)


@patch("src.haystack_chat_agent._rag_service")
def test_get_document_details_tool(mock_rag):
    mock_rag.get_document_details_formatted.return_value = "Doc details"
    result = get_document_details.function(document_id=5)
    assert result == "Doc details"
    mock_rag.get_document_details_formatted.assert_called_once_with(document_id=5)


def test_build_agent_instantiation():
    agent = build_agent()
    assert agent is not None
    assert agent.chat_generator is not None
