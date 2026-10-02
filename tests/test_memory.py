from pathlib import Path

from atlas_rag.memory import ConversationMemory


def test_memory_round_trip_and_limit(tmp_path: Path) -> None:
    memory = ConversationMemory(tmp_path / "memory.sqlite3", max_turns=2)
    memory.add("session", "user", "one")
    memory.add("session", "assistant", "two")
    memory.add("session", "user", "three")
    memory.add("session", "assistant", "four")
    memory.add("session", "user", "five")
    memory.add("session", "assistant", "six")
    assert memory.get("session") == [
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
        {"role": "user", "content": "five"},
        {"role": "assistant", "content": "six"},
    ]


def test_memory_clear(tmp_path: Path) -> None:
    memory = ConversationMemory(tmp_path / "memory.sqlite3")
    memory.add("session", "user", "hello")
    memory.clear("session")
    assert memory.get("session") == []
