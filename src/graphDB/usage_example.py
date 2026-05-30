"""

Demonstrates how to use EntityGraphMemory with Neo4j.
"""

from src.graphDB import EntityGraphMemory


def main():
    memory = EntityGraphMemory(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="your_password_here",
    )
    memory.write_entity(
        name="Alice",
        entity_type="PERSON",
        properties={"role": "Engineer", "department": "Backend"},
        thread_id="thread_001",
    )

    memory.write_entity(
        name="Acme Corp",
        entity_type="ORGANIZATION",
        properties={"industry": "Tech", "location": "San Francisco"},
    )
    memory.write_relationship(
        source="Alice",
        target="Acme Corp",
        rel_type="WORKS_AT",
        properties={"since": "2023-01-15"},
    )

    text = """
    John is a data scientist at Google. He works closely with Sarah,
    who is the VP of Engineering. They are building a new ML platform
    that integrates with TensorFlow and PyTorch.
    """

    result = memory.write_entities_from_text(text, thread_id="thread_002")
    print(f"Extracted {result['entities_written']} entities and {result['relationships_written']} relationships")

    alice = memory.search_entity("Alice")
    print(f"Found: {alice}")

    # Search by type
    people = memory.search_entity("", entity_type="PERSON")
    print(f"People: {people}")

    alice_rels = memory.get_related_entities("Alice")
    print(f"Alice's relationships: {alice_rels}")

    # Get 2-hop neighbors
    extended = memory.get_entity_relationships("Alice", depth=2)
    print(f"Alice's extended network: {extended}")

    from langchain_core.documents import Document

    memory.write_entity_documents(
        texts=["Bob is a product manager", "Carol is a designer"],
        metadatas=[
            {"name": "Bob", "type": "PERSON"},
            {"name": "Carol", "type": "PERSON"},
        ],
        thread_id="thread_003",
    )

    # Search and get Documents back (compatible with existing search_entity signature)
    docs = memory.search_entity_documents("Bob", k=5)
    for doc in docs:
        print(f"Document: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")

    # ─────────────────────────────────────────────────────────────────────
    # 8. Stats
    # ─────────────────────────────────────────────────────────────────────
    stats = memory.get_stats()
    print(f"Graph stats: {stats}")

    # ─────────────────────────────────────────────────────────────────────
    # 9. Cleanup
    # ─────────────────────────────────────────────────────────────────────
    memory.close()


if __name__ == "__main__":
    main()
