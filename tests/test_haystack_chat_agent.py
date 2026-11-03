"""Tests for Haystack Chat Agent integration."""

from unittest.mock import MagicMock, patch

from haystack.dataclasses import ChatMessage, StreamingChunk

from src.haystack_chat_agent import (
    TOOLS,
    add_numbers,
    build_agent,
    chat_loop,
    get_current_time,
    get_document_details,
    list_knowledge_base_documents,
    search_knowledge_base,
    stream_to_console,
)


def test_tools_registered():
    tool_names = [t.name for t in TOOLS]
    assert "search_knowledge_base" in tool_names
    assert "list_knowledge_base_documents" in tool_names
    assert "get_document_details" in tool_names
    assert "get_current_time" in tool_names
    assert "add_numbers" in tool_names


def test_get_current_time_tool():
    result = get_current_time.function()
    assert isinstance(result, str)
    assert "T" in result


def test_add_numbers_tool():
    assert add_numbers.function(a=3.5, b=4.5) == 8.0


@patch("src.haystack_chat_agent._rag_service")
def test_search_knowledge_base_tool(mock_rag):
    mock_rag.search_and_format.return_value = "Mocked KB Results"
    result = search_knowledge_base.function(query="architecture guidelines", limit=5)
    assert result == "Mocked KB Results"
    mock_rag.search_and_format.assert_called_once_with(
        query="architecture guidelines", limit=5
    )


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


def test_build_agent_custom_args():
    custom_tools = [get_current_time]
    agent = build_agent(
        model="custom-model",
        tools=custom_tools,
        system_prompt="Custom System Prompt",
    )
    assert agent is not None
    assert agent.system_prompt == "Custom System Prompt"


def test_stream_to_console(capsys):
    chunk = StreamingChunk(content="Hello ")
    stream_to_console(chunk)
    captured = capsys.readouterr()
    assert "Hello " in captured.out


def test_stream_to_console_tool_call(capsys):
    tc = MagicMock(tool_name="search_knowledge_base")
    chunk = MagicMock(content=None, tool_calls=[tc], tool_call_result=None)
    stream_to_console(chunk)
    captured = capsys.readouterr()
    assert "[calling tool: search_knowledge_base]" in captured.out


def test_stream_to_console_tool_result(capsys):
    tr = MagicMock(result="Found 1 document")
    chunk = MagicMock(content=None, tool_calls=None, tool_call_result=tr)
    stream_to_console(chunk)
    captured = capsys.readouterr()
    assert "[tool result: Found 1 document]" in captured.out


def test_chat_loop_exit(capsys):
    mock_agent = MagicMock()
    with patch("builtins.input", side_effect=["exit"]):
        chat_loop(mock_agent)
    captured = capsys.readouterr()
    assert "Chat started." in captured.out
    assert "Bye!" in captured.out
    mock_agent.run.assert_not_called()


def test_chat_loop_reset_and_query(capsys):
    mock_agent = MagicMock()
    mock_agent.run.return_value = {
        "messages": [
            ChatMessage.from_user("hello"),
            ChatMessage.from_assistant("Hi there!"),
        ]
    }
    with patch("builtins.input", side_effect=["/reset", "hello", "quit"]):
        chat_loop(mock_agent)
    captured = capsys.readouterr()
    assert "(history cleared)" in captured.out
    assert "Bye!" in captured.out
    mock_agent.run.assert_called_once()


def test_chat_loop_handles_exception(capsys):
    mock_agent = MagicMock()
    mock_agent.run.side_effect = Exception("API error")
    with patch("builtins.input", side_effect=["hello", "exit"]):
        chat_loop(mock_agent)
    captured = capsys.readouterr()
    assert "[error] API error" in captured.err
    assert "Bye!" in captured.out
