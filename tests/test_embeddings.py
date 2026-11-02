"""Tests for embedding generation module."""

from unittest.mock import MagicMock, patch
from src.embeddings import DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL, embed_query, embed_texts, get_openai_client


def test_default_embedding_dimension():
    assert DEFAULT_EMBED_DIM == 2048


def test_embed_texts_empty():
    assert embed_texts([]) == []


@patch("src.embeddings.get_openai_client")
def test_embed_texts_default_dimension(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    item1 = MagicMock(index=0, embedding=[0.1, 0.2, 0.3])
    item2 = MagicMock(index=1, embedding=[0.4, 0.5, 0.6])
    mock_client.embeddings.create.return_value = MagicMock(data=[item2, item1])

    result = embed_texts(["first", "second"])
    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]
    mock_client.embeddings.create.assert_called_once_with(
        input=["first", "second"],
        model=DEFAULT_EMBED_MODEL,
        dimensions=2048,
    )


@patch("src.embeddings.get_openai_client")
def test_embed_texts_dimension_fallback(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # First call with dimensions raises, second call without dimensions succeeds
    item = MagicMock(index=0, embedding=[0.7, 0.8, 0.9])
    mock_client.embeddings.create.side_effect = [
        Exception("dimensions parameter not supported"),
        MagicMock(data=[item]),
    ]

    result = embed_texts(["fallback text"])
    assert result == [[0.7, 0.8, 0.9]]
    assert mock_client.embeddings.create.call_count == 2
    mock_client.embeddings.create.assert_any_call(
        input=["fallback text"],
        model=DEFAULT_EMBED_MODEL,
        dimensions=2048,
    )
    mock_client.embeddings.create.assert_called_with(
        input=["fallback text"],
        model=DEFAULT_EMBED_MODEL,
    )


@patch("src.embeddings.embed_texts")
def test_embed_query_default_dimension(mock_embed_texts):
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
    res = embed_query("hello")
    assert res == [0.1, 0.2, 0.3]
    mock_embed_texts.assert_called_once_with(["hello"], model=DEFAULT_EMBED_MODEL, dimensions=2048)


@patch("src.embeddings.embed_texts")
def test_embed_query_empty_result(mock_embed_texts):
    mock_embed_texts.return_value = []
    res = embed_query("hello")
    assert res == []
