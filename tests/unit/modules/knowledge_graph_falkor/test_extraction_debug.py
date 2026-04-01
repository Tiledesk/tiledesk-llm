"""
Debug script to test GraphRAG extraction with relationship types
"""
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Sample text for debt collection
SAMPLE_TEXT = """
On 2023-03-15, Banca ABC granted a €250,000 mortgage loan (ref: LOAN-789) to Marco Rossi.
The loan was guaranteed by a suretyship from Ditta Rossi SRL.
After three missed payments in Q4 2024, the bank declared default on 2024-11-30
and initiated foreclosure proceedings (case #FC-2024-88) on 2025-01-10.
A partial payment of €15,000 was received on 2025-02-20.
"""


async def test_extraction():
    """Test the extraction and see what comes out"""
    from tilellm.modules.knowledge_graph_falkor.tools.graphrag_extractor import (
        GraphRAGExtractor,
        handle_single_relationship_extraction,
        split_string_by_multi_markers
    )
    from tilellm.modules.knowledge_graph_falkor.tools.extraction_prompts import (DEFAULT_ENTITY_TYPES_GENERIC)

    # Mock LLM that returns example output
    class MockLLM:
        async def ainvoke(self, messages):
            # Simulate LLM response in correct format
            response_text = '''("entity","Banca ABC","organization","Italian bank acting as creditor in mortgage loan LOAN-789")
("entity","Marco Rossi","person","DEBTOR - Borrower of €250,000 mortgage loan LOAN-789")
("entity","Ditta Rossi SRL","organization","Company providing suretyship guarantee for Marco Rossi's loan")
("entity","LOAN-789","loan","€250,000 mortgage loan originated 2023-03-15, secured by real estate, currently in default")
("entity","Suretyship for LOAN-789","guarantee","Personal guarantee provided by Ditta Rossi SRL for full loan amount")
("entity","Default on LOAN-789","default","Formal default declared 2024-11-30 after three consecutive missed payments in Q4 2024")
("entity","Foreclosure FC-2024-88","legal_proceeding","Enforcement action initiated 2025-01-10 following loan default")
("entity","Partial payment €15k","payment","Settlement payment of €15,000 received 2025-02-20 during foreclosure proceedings")
("relationship","Banca ABC","LOAN-789","HAS_LOAN","Banca ABC originated and owns mortgage loan LOAN-789","9","2023-03-15")
("relationship","Marco Rossi","LOAN-789","OBLIGATED_UNDER","Marco Rossi is primary borrower obligated under loan LOAN-789","10","2023-03-15")
("relationship","Ditta Rossi SRL","LOAN-789","GUARANTEES","Ditta Rossi SRL executed suretyship guarantee for LOAN-789","9","2023-03-15")
("relationship","LOAN-789","Suretyship for LOAN-789","SECURED_BY","Loan LOAN-789 is secured by suretyship guarantee","10","2023-03-15")
("relationship","LOAN-789","Default on LOAN-789","RESULTED_IN","Loan LOAN-789 entered formal default status after payment failures","10","2024-11-30")
("relationship","Default on LOAN-789","Foreclosure FC-2024-88","TRIGGERED","Default directly triggered initiation of foreclosure proceedings","10","2025-01-10")
("relationship","LOAN-789","Foreclosure FC-2024-88","HAS_LEGAL_ACTION","Loan has active foreclosure proceedings","10","2025-01-10")
("relationship","LOAN-789","Partial payment €15k","HAS_PAYMENT","Partial payment received during active foreclosure proceedings","7","2025-02-20")
[COMPLETED]'''

            class Response:
                content = response_text

            return Response()

        async def __call__(self, prompt):
            """Make the mock callable as well"""
            result = await self.ainvoke([])
            return result.content

    # Test with mock LLM
    print("=" * 60)
    print("Testing GraphRAG Extraction with Mock LLM")
    print("=" * 60)

    extractor = GraphRAGExtractor(llm_invoker=MockLLM(), entity_types=DEFAULT_ENTITY_TYPES_GENERIC)

    entities, relationships, token_count = await extractor.extract_chunk("test_chunk_1", SAMPLE_TEXT)

    print("\n--- ENTITIES EXTRACTED ---")
    for entity_name, entity_list in entities.items():
        for entity in entity_list:
            print(f"  - {entity['entity_name']} ({entity['entity_type']}): {entity['description'][:50]}...")

    print("\n--- RELATIONSHIPS EXTRACTED ---")
    for rel_key, rel_list in relationships.items():
        for rel in rel_list:
            rel_type = rel.get('relationship_type', 'UNKNOWN')
            print(f"  - {rel['src_id']} -[{rel_type}]-> {rel['tgt_id']}")
            print(f"    Description: {rel.get('description', 'N/A')[:60]}...")
            print(f"    Weight: {rel.get('weight', 1.0)}, Date: {rel.get('date', 'N/A')}")

    # Test individual parsing
    print("\n" + "=" * 60)
    print("Testing Individual Relationship Parsing")
    print("=" * 60)

    test_cases = [
        '("relationship","Banca ABC","LOAN-789","HAS_LOAN","Banca ABC owns loan","9","2023-03-15")',
        '("relationship","Marco Rossi","LOAN-789","OBLIGATED_UNDER","Primary borrower","10","2023-03-15")',
        '("relationship","LOAN-789","Default-001","RESULTED_IN","Loan defaulted","10","2024-11-30")',
    ]

    for test_case in test_cases:
        print(f"\nInput: {test_case}")
        # Remove surrounding parentheses
        if test_case.startswith('(') and test_case.endswith(')'):
            test_case = test_case[1:-1]

        record_attributes = split_string_by_multi_markers(test_case, ['","'])
        print(f"Parsed fields: {record_attributes}")

        rel = handle_single_relationship_extraction(record_attributes, "test_chunk")
        if rel:
            print(f"Result: {rel['src_id']} -[{rel['relationship_type']}]-> {rel['tgt_id']}")
            print(f"  Weight: {rel['weight']}, Date: {rel.get('date', 'N/A')}")
        else:
            print("  FAILED TO PARSE")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_extraction())
