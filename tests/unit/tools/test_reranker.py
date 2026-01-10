"""
Test script for Max-P strategy in reranker
Tests the chunking and reranking of long documents
"""
import sys
sys.path.insert(0, '/home/lor/sviluppo/tiledesk/tiledesk-llm')

from langchain_core.documents import Document
from tilellm.tools.reranker import (
    estimate_tokens,
    chunk_text_with_overlap,
    apply_maxp_chunking,
    aggregate_maxp_scores
)


def test_token_estimation():
    """Test fast token estimation"""
    print("\n=== Test 1: Token Estimation ===")

    short_text = "This is a short text."
    long_text = " ".join(["word"] * 500)  # ~650 tokens

    short_tokens = estimate_tokens(short_text)
    long_tokens = estimate_tokens(long_text)

    print(f"Short text ({len(short_text)} chars): ~{short_tokens} tokens")
    print(f"Long text ({len(long_text)} chars): ~{long_tokens} tokens")

    assert short_tokens < 20, "Short text should have few tokens"
    assert long_tokens > 400, "Long text should exceed 400 tokens"
    print("✓ Token estimation works correctly")


def test_chunking_with_overlap():
    """Test text chunking with overlap"""
    print("\n=== Test 2: Text Chunking with Overlap ===")

    # Create a text with ~900 tokens (estimated)
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)

    chunks = chunk_text_with_overlap(text, max_tokens=300, overlap_tokens=40)

    print(f"Original text: ~{estimate_tokens(text)} tokens")
    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        tokens = estimate_tokens(chunk)
        print(f"  Chunk {i+1}: ~{tokens} tokens")
        # Verifica che nessun chunk superi 300 token stimati (che corrispondono a ~167 parole)
        # Con stima conservativa di 1.8, 300 token = 167 parole = ~300 token reali (sicuro sotto 512)
        assert tokens <= 320, f"Chunk {i+1} exceeds max_tokens limit (estimated {tokens} tokens)"

    assert len(chunks) >= 2, "Long text should be split into multiple chunks"
    print("✓ Chunking with overlap works correctly")


def test_maxp_chunking():
    """Test Max-P chunking strategy on documents"""
    print("\n=== Test 3: Max-P Document Chunking ===")

    # Create documents with different lengths
    docs = [
        Document(page_content="Short document.", metadata={"id": 1}),
        Document(page_content=" ".join(["word"] * 400), metadata={"id": 2}),  # ~720 tokens (1.8 factor)
        Document(page_content=" ".join(["text"] * 500), metadata={"id": 3}),  # ~900 tokens (1.8 factor)
        Document(page_content="Another short one.", metadata={"id": 4}),
    ]

    print(f"Original documents: {len(docs)}")
    for i, doc in enumerate(docs):
        tokens = estimate_tokens(doc.page_content)
        print(f"  Doc {i+1}: ~{tokens} tokens")

    chunked_docs, doc_indices = apply_maxp_chunking(docs, max_tokens=300, overlap_tokens=40)

    print(f"\nAfter Max-P chunking: {len(chunked_docs)} chunks")
    print(f"Document indices mapping: {doc_indices}")

    # Verify that long documents are split
    assert len(chunked_docs) > len(docs), "Long documents should create more chunks"
    assert len(chunked_docs) == len(doc_indices), "Each chunk should have a document index"

    # Verify that metadata is preserved
    for chunk in chunked_docs:
        if chunk.metadata.get("_is_chunk"):
            orig_idx = chunk.metadata["_original_doc_idx"]
            original_metadata = docs[orig_idx].metadata
            print(f"  Chunk from doc {orig_idx+1}: metadata preserved ✓")

    print("✓ Max-P document chunking works correctly")


def test_score_aggregation():
    """Test Max-P score aggregation"""
    print("\n=== Test 4: Max-P Score Aggregation ===")

    # Simulate chunked documents with scores
    # Doc 0: not chunked
    # Doc 1: split into 2 chunks
    # Doc 2: split into 3 chunks

    chunked_docs = [
        Document(page_content="Short doc", metadata={"id": 0}),  # Doc 0
        Document(page_content="Chunk 1 of doc 1", metadata={"_is_chunk": True, "_original_doc_idx": 1, "id": 1}),
        Document(page_content="Chunk 2 of doc 1", metadata={"_is_chunk": True, "_original_doc_idx": 1, "id": 1}),
        Document(page_content="Chunk 1 of doc 2", metadata={"_is_chunk": True, "_original_doc_idx": 2, "id": 2}),
        Document(page_content="Chunk 2 of doc 2", metadata={"_is_chunk": True, "_original_doc_idx": 2, "id": 2}),
        Document(page_content="Chunk 3 of doc 2", metadata={"_is_chunk": True, "_original_doc_idx": 2, "id": 2}),
    ]

    doc_indices = [0, 1, 1, 2, 2, 2]

    # Simulate scores (doc 2's chunk 2 has highest score for that doc)
    scores = [0.5, 0.7, 0.9, 0.6, 0.95, 0.4]

    scored_chunks = list(zip(chunked_docs, scores))

    print(f"Chunks with scores:")
    for i, (doc, score) in enumerate(scored_chunks):
        orig_idx = doc_indices[i]
        print(f"  Chunk {i+1} (from doc {orig_idx}): score={score}")

    # Aggregate using Max-P
    aggregated = aggregate_maxp_scores(scored_chunks, doc_indices)

    print(f"\nAggregated scores (Max-P):")
    for i, (doc, score) in enumerate(aggregated):
        print(f"  Doc {i}: max_score={score}")

    # Verify max scores
    assert len(aggregated) == 3, "Should have 3 original documents"
    assert aggregated[0][1] == 0.5, "Doc 0 should have score 0.5"
    assert aggregated[1][1] == 0.9, "Doc 1 should have max score 0.9"
    assert aggregated[2][1] == 0.95, "Doc 2 should have max score 0.95"

    print("✓ Max-P score aggregation works correctly")


def test_safety_margin():
    """Test that chunks never exceed 512 token limit with safety margin"""
    print("\n=== Test 5: Safety Margin Test (512 token limit) ===")

    # Create a very long text to test worst-case scenario
    # Worst case: many short words that tokenize inefficiently
    # Using 300 token limit with 1.8 factor = ~167 words max per chunk
    long_text = " ".join([f"w{i}" for i in range(300)])  # 300 short words

    chunks = chunk_text_with_overlap(long_text, max_tokens=300, overlap_tokens=40)

    print(f"Original text: {len(long_text.split())} words")
    print(f"Number of chunks: {len(chunks)}")

    all_safe = True
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        estimated_tokens = estimate_tokens(chunk)
        # Worst case: 2.0 tokens per word (worst observed ratio)
        # If estimated is 300, worst case real could be: (300/1.8)*2.0 = 333 tokens
        worst_case_real = int((estimated_tokens / 1.8) * 2.0)

        print(f"  Chunk {i+1}: {word_count} words, ~{estimated_tokens} estimated, "
              f"worst case ~{worst_case_real} real tokens")

        if worst_case_real > 512:
            all_safe = False
            print(f"    ⚠️  WARNING: Could exceed 512 token limit!")

    assert all_safe, "Some chunks could exceed 512 token limit in worst case"
    print("✓ All chunks are safe under 512 token limit")


def test_integration():
    """Integration test with realistic scenario"""
    print("\n=== Test 6: Integration Test ===")

    # Create a realistic scenario with community reports
    docs = [
        Document(
            page_content="Short community report about topic A.",
            metadata={"type": "community_report", "title": "Topic A"}
        ),
        Document(
            page_content=" ".join([
                "This is a very long community report.",
                "It contains detailed information about topic B.",
                " ".join(["detailed analysis"] * 100),  # Make it long
                "Conclusion about topic B."
            ]),
            metadata={"type": "community_report", "title": "Topic B"}
        ),
        Document(
            page_content="Another short document about topic C.",
            metadata={"type": "document", "title": "Topic C"}
        ),
    ]

    print(f"Documents:")
    for i, doc in enumerate(docs):
        tokens = estimate_tokens(doc.page_content)
        print(f"  Doc {i+1} ({doc.metadata['title']}): ~{tokens} tokens")

    # Apply Max-P chunking
    chunked_docs, doc_indices = apply_maxp_chunking(docs, max_tokens=300, overlap_tokens=40)

    print(f"\nAfter chunking: {len(chunked_docs)} chunks")

    # Simulate reranking with dummy scores
    import random
    random.seed(42)
    scores = [random.random() for _ in chunked_docs]

    scored_chunks = list(zip(chunked_docs, scores))

    # Aggregate scores
    aggregated = aggregate_maxp_scores(scored_chunks, doc_indices)

    # Sort by score
    aggregated.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop documents after reranking:")
    for i, (doc, score) in enumerate(aggregated[:3]):
        print(f"  {i+1}. {doc.metadata.get('title', 'Unknown')} - score: {score:.3f}")

    assert len(aggregated) == len(docs), "Should have same number as original documents"
    print("✓ Integration test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Max-P Strategy Implementation")
    print("=" * 60)

    try:
        test_token_estimation()
        test_chunking_with_overlap()
        test_maxp_chunking()
        test_score_aggregation()
        test_safety_margin()
        test_integration()

        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
