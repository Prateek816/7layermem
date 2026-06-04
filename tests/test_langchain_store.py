from unittest.mock import MagicMock, call

import pytest
from langchain_core.documents import Document

from src.langchain_extension.store import SevenLayerStore


@pytest.fixture
def mock_agent_memory():
    return MagicMock()


@pytest.fixture
def store(mock_agent_memory):
    return SevenLayerStore(mock_agent_memory, store_name="knowledge")


class TestInit:
    def test_valid_store_name(self, mock_agent_memory):
        s = SevenLayerStore(mock_agent_memory, store_name="knowledge")
        assert s._store_name == "knowledge"

    def test_all_valid_names(self, mock_agent_memory):
        for name in ("knowledge", "entity", "workflow", "summary"):
            s = SevenLayerStore(mock_agent_memory, store_name=name)
            assert s._store_name == name

    def test_invalid_store_name(self, mock_agent_memory):
        with pytest.raises(ValueError, match="store_name must be one of"):
            SevenLayerStore(mock_agent_memory, store_name="invalid")


class TestMget:
    def test_get_returns_content(self, store, mock_agent_memory):
        mock_agent_memory._mm.search_store.return_value = [
            Document(page_content="value", metadata={"key": "mykey"})
        ]
        result = store.mget(["mykey"])
        assert result == ["value"]

    def test_get_returns_none_when_not_found(self, store, mock_agent_memory):
        mock_agent_memory._mm.search_store.return_value = []
        result = store.mget(["missing"])
        assert result == [None]

    def test_get_multiple_keys(self, store, mock_agent_memory):
        mock_agent_memory._mm.search_store.side_effect = [
            [Document(page_content="v1", metadata={"key": "k1"})],
            [],
        ]
        result = store.mget(["k1", "k2"])
        assert result == ["v1", None]


class TestMset:
    def test_set_calls_write_to_store(self, store, mock_agent_memory):
        store.mset([("mykey", "myvalue")])
        mock_agent_memory._mm.write_to_store.assert_called_once_with(
            "knowledge",
            texts=["myvalue"],
            metadatas=[{"key": "mykey"}],
            ids=[mock_agent_memory._mm.write_to_store.call_args.kwargs["ids"][0]],
        )

    def test_set_multiple_pairs(self, store, mock_agent_memory):
        store.mset([("k1", "v1"), ("k2", "v2")])
        assert mock_agent_memory._mm.write_to_store.call_count == 2


class TestMdelete:
    def test_delete_finds_and_removes(self, store, mock_agent_memory):
        doc = MagicMock()
        doc.id = "doc_abc"
        mock_agent_memory._mm.search_store.return_value = [doc]
        store.mdelete(["mykey"])
        mock_agent_memory._mm.delete_from_store.assert_called_once_with(
            "knowledge", ["doc_abc"]
        )

    def test_delete_no_match(self, store, mock_agent_memory):
        mock_agent_memory._mm.search_store.return_value = []
        store.mdelete(["missing"])
        mock_agent_memory._mm.delete_from_store.assert_not_called()


class TestYieldKeys:
    def test_yield_all_keys(self, store, mock_agent_memory):
        mock_agent_memory._mm.search_store.return_value = [
            Document(page_content="a", metadata={"key": "alpha"}),
            Document(page_content="b", metadata={"key": "beta"}),
        ]
        keys = list(store.yield_keys())
        assert keys == ["alpha", "beta"]

    def test_yield_keys_with_prefix(self, store, mock_agent_memory):
        mock_agent_memory._mm.search_store.return_value = [
            Document(page_content="a", metadata={"key": "alpha"}),
            Document(page_content="b", metadata={"key": "beta"}),
            Document(page_content="c", metadata={"key": "alpine"}),
        ]
        keys = list(store.yield_keys(prefix="al"))
        assert keys == ["alpha", "alpine"]

    def test_yield_keys_empty_metadata(self, store, mock_agent_memory):
        mock_agent_memory._mm.search_store.return_value = [
            Document(page_content="a", metadata={}),
        ]
        keys = list(store.yield_keys())
        assert keys == [""]
