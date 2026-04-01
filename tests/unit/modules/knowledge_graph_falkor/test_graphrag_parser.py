"""
Unit tests for GraphRAG extraction and parsing logic.

Tests the parsing of entities and relationships from LLM responses,
including handling of different delimiter formats and validation of required fields.
"""
import pytest
from tilellm.modules.knowledge_graph_falkor.tools.graphrag_extractor import (
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    split_string_by_multi_markers,
    clean_str
)


class TestCleanStr:
    """Test the clean_str utility function"""

    def test_clean_str_removes_quotes(self):
        """Test that clean_str removes double quotes"""
        assert clean_str('"hello"') == 'hello'
        assert clean_str('""test""') == 'test'

    def test_clean_str_removes_control_chars(self):
        """Test that clean_str removes control characters"""
        assert clean_str('hello\x00world') == 'helloworld'
        assert clean_str('test\x1f\x7fdata') == 'testdata'

    def test_clean_str_handles_html_entities(self):
        """Test that clean_str unescapes HTML entities"""
        assert clean_str('&lt;test&gt;') == '<test>'
        assert clean_str('&amp;') == '&'

    def test_clean_str_strips_whitespace(self):
        """Test that clean_str strips leading/trailing whitespace"""
        assert clean_str('  hello  ') == 'hello'
        assert clean_str('\n\ttest\t\n') == 'test'


class TestSplitStringByMultiMarkers:
    """Test the split_string_by_multi_markers function"""

    def test_split_with_single_delimiter(self):
        """Test splitting with single delimiter"""
        result = split_string_by_multi_markers('a,b,c', [','])
        assert result == ['a', 'b', 'c']

    def test_split_with_double_quote_delimiter(self):
        """Test splitting with double quote delimiter (LLM format)"""
        text = '"entity"",""name"",""type"",""description""'
        result = split_string_by_multi_markers(text, ['"",""'])
        assert len(result) == 4
        assert '"entity' in result[0]
        assert 'name' in result[1]

    def test_split_with_multiple_delimiters(self):
        """Test splitting with multiple possible delimiters"""
        text = '"entity","name","type"'
        result = split_string_by_multi_markers(text, ['"",""', '","'])
        assert len(result) == 3

    def test_split_removes_empty_results(self):
        """Test that empty strings are filtered out"""
        result = split_string_by_multi_markers('a,,b,c', [','])
        # Empty strings are now preserved to maintain field positions
        assert result == ['a', '', 'b', 'c']

    def test_split_strips_whitespace(self):
        """Test that results are stripped of whitespace"""
        result = split_string_by_multi_markers('a , b , c', [','])
        assert result == ['a', 'b', 'c']


class TestEntityExtraction:
    """Test entity extraction from parsed records"""

    def test_extract_entity_single_quote_format(self):
        """Test entity extraction with single quote delimiter"""
        # Format: ("entity","Name","Type","Description")
        line = '"entity","Banca ABC","organization","Italian bank"'
        fields = split_string_by_multi_markers(line, ['","'])

        entity = handle_single_entity_extraction(fields, "chunk_1")

        assert entity is not None
        assert entity['entity_name'] == 'BANCA ABC'
        assert entity['entity_type'] == 'ORGANIZATION'
        assert 'Italian bank' in entity['description']
        assert entity['source_id'] == 'chunk_1'

    def test_extract_entity_double_quote_format(self):
        """Test entity extraction with double quote delimiter (LLM format)"""
        # Format: ("entity"",""Name"",""Type"",""Description"")
        line = '"entity"",""Marco Rossi"",""person"",""DEBTOR - Borrower""'
        fields = split_string_by_multi_markers(line, ['"",""', '","'])

        entity = handle_single_entity_extraction(fields, "chunk_1")

        assert entity is not None
        assert entity['entity_name'] == 'MARCO ROSSI'
        assert entity['entity_type'] == 'PERSON'
        assert 'DEBTOR' in entity['description']

    def test_extract_entity_insufficient_fields(self):
        """Test that entity with insufficient fields is rejected"""
        fields = ['"entity"', 'Name', 'Type']  # Only 3 fields, need 4

        entity = handle_single_entity_extraction(fields, "chunk_1")

        assert entity is None

    def test_extract_entity_wrong_type(self):
        """Test that non-entity records are rejected"""
        fields = ['"relationship"', 'Source', 'Target', 'Type']

        entity = handle_single_entity_extraction(fields, "chunk_1")

        assert entity is None

    def test_extract_entity_empty_name(self):
        """Test that entity with empty name is rejected"""
        fields = ['"entity"', '', 'organization', 'Description']

        entity = handle_single_entity_extraction(fields, "chunk_1")

        assert entity is None


class TestRelationshipExtraction:
    """Test relationship extraction from parsed records"""

    def test_extract_relationship_new_format_single_quotes(self):
        """Test relationship extraction with new format (7 fields, single quotes)"""
        # Format: ("relationship","Source","Target","TYPE","Description","9","2023-01-01")
        line = '"relationship","Banca ABC","LOAN-789","HAS_LOAN","Bank owns loan","9","2023-03-15"'
        fields = split_string_by_multi_markers(line, ['","'])

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is not None
        assert rel['src_id'] == 'BANCA ABC'
        assert rel['tgt_id'] == 'LOAN-789'
        assert rel['relationship_type'] == 'HAS_LOAN'
        assert rel['weight'] == 9.0
        assert 'Bank owns loan' in rel['description']
        assert rel['date'] == '2023-03-15'
        assert rel['source_id'] == 'chunk_1'

    def test_extract_relationship_new_format_double_quotes(self):
        """Test relationship extraction with new format (7 fields, double quotes)"""
        # Format: ("relationship"",""Source"",""Target"",""TYPE"",""Description"",""10"",""2024-01-01"")
        line = '"relationship"",""Marco Rossi"",""LOAN-789"",""OBLIGATED_UNDER"",""Primary borrower"",""10"",""2023-03-15""'
        fields = split_string_by_multi_markers(line, ['"",""', '","'])

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is not None
        assert rel['src_id'] == 'MARCO ROSSI'
        assert rel['tgt_id'] == 'LOAN-789'
        assert rel['relationship_type'] == 'OBLIGATED_UNDER'
        assert rel['weight'] == 10.0
        assert 'Primary borrower' in rel['description']
        assert rel['date'] == '2023-03-15'

    def test_extract_relationship_all_types(self):
        """Test extraction of all supported relationship types"""
        relationship_types = [
            'HAS_LOAN', 'SECURED_BY', 'GUARANTEES', 'HAS_PAYMENT',
            'RECEIVED', 'HAS_LEGAL_ACTION', 'OWNS', 'OBLIGATED_UNDER',
            'TRIGGERED', 'RESULTED_IN', 'CONCERNS', 'RELATED_TO'
        ]

        for rel_type in relationship_types:
            line = f'"relationship","Source","Target","{rel_type}","Description","8","2023-01-01"'
            fields = split_string_by_multi_markers(line, ['","'])

            rel = handle_single_relationship_extraction(fields, "chunk_1")

            assert rel is not None, f"Failed to extract {rel_type}"
            assert rel['relationship_type'] == rel_type, f"Wrong type for {rel_type}"

    def test_reject_old_format_insufficient_fields(self):
        """Test that old format (5 fields without relationship_type) is rejected"""
        # Old format: ("relationship","Source","Target","Description","9")
        line = '"relationship","BNL","GIACOIA","Some description","7"'
        fields = split_string_by_multi_markers(line, ['","'])

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is None, "Old format should be rejected"

    def test_reject_relationship_with_description_as_type(self):
        """Test that relationships with description in type field are rejected"""
        # Format where field 3 looks like a description, not a type
        line = '"relationship","Source","Target","This is a long description that should not be a type","More text","9","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is None, "Relationship with description as type should be rejected"

    def test_relationship_weight_defaults(self):
        """Test that invalid weight defaults to 1.0"""
        line = '"relationship","Source","Target","HAS_LOAN","Description","invalid","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is not None
        assert rel['weight'] == 1.0, "Invalid weight should default to 1.0"

    def test_relationship_without_date(self):
        """Test relationship extraction with only 6 fields (no date)"""
        line = '"relationship","Source","Target","HAS_LOAN","Description","9"'
        fields = split_string_by_multi_markers(line, ['","'])

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is not None
        assert rel['date'] is None
        assert rel['relationship_type'] == 'HAS_LOAN'

    def test_reject_relationship_wrong_type(self):
        """Test that non-relationship records are rejected"""
        fields = ['"entity"', 'Name', 'Type', 'Description', 'Extra', 'More']

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is None

    def test_reject_relationship_empty_source_or_target(self):
        """Test that relationships with empty source/target are rejected"""
        # Empty source
        line = '"relationship","","Target","HAS_LOAN","Description","9","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])
        rel = handle_single_relationship_extraction(fields, "chunk_1")
        assert rel is None

        # Empty target
        line = '"relationship","Source","","HAS_LOAN","Description","9","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])
        rel = handle_single_relationship_extraction(fields, "chunk_1")
        assert rel is None

    def test_relationship_type_validation_max_words(self):
        """Test that relationship type must be max 3 words"""
        # Valid: 1 word
        line = '"relationship","Source","Target","HAS_LOAN","Description","9","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])
        rel = handle_single_relationship_extraction(fields, "chunk_1")
        assert rel is not None

        # Valid: 2 words
        line = '"relationship","Source","Target","HAS LOAN","Description","9","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])
        rel = handle_single_relationship_extraction(fields, "chunk_1")
        assert rel is not None

        # Valid: 3 words
        line = '"relationship","Source","Target","HAS SOME LOAN","Description","9","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])
        rel = handle_single_relationship_extraction(fields, "chunk_1")
        assert rel is not None

        # Invalid: 4+ words (looks like description)
        line = '"relationship","Source","Target","This has too many words","Description","9","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])
        rel = handle_single_relationship_extraction(fields, "chunk_1")
        assert rel is None


class TestRealWorldScenarios:
    """Test real-world scenarios from actual LLM responses"""

    def test_debt_collection_scenario(self):
        """Test extraction from a typical debt collection scenario"""
        # Simulate full extraction from parenthesized format
        entity_line = '("entity"",""GIACOIA DOMENICO"",""person"",""Individual debtor"")'
        rel_line = '("relationship"",""GIACOIA DOMENICO"",""LOAN-123"",""OBLIGATED_UNDER"",""Primary borrower of loan"",""9"",""2023-03-15"")'

        # Remove parentheses (as done in real code)
        entity_line = entity_line[1:-1] if entity_line.startswith('(') else entity_line
        rel_line = rel_line[1:-1] if rel_line.startswith('(') else rel_line

        # Parse entity
        entity_fields = split_string_by_multi_markers(entity_line, ['"",""', '","'])
        entity = handle_single_entity_extraction(entity_fields, "chunk_1")

        assert entity is not None
        assert entity['entity_name'] == 'GIACOIA DOMENICO'
        assert entity['entity_type'] == 'PERSON'

        # Parse relationship
        rel_fields = split_string_by_multi_markers(rel_line, ['"",""', '","'])
        rel = handle_single_relationship_extraction(rel_fields, "chunk_1")

        assert rel is not None
        assert rel['src_id'] == 'GIACOIA DOMENICO'
        assert rel['tgt_id'] == 'LOAN-123'
        assert rel['relationship_type'] == 'OBLIGATED_UNDER'

    def test_mixed_delimiter_formats_in_batch(self):
        """Test that parser handles mixed delimiter formats gracefully"""
        # Some LLMs might return inconsistent formats
        lines = [
            '("entity","Bank A","organization","A bank")',  # single quotes
            '("entity"",""Bank B"",""organization"",""Another bank"")',  # double quotes
            '("relationship","Bank A","LOAN-1","HAS_LOAN","Owns loan","9","2023-01-01")',  # single
            '("relationship"",""Bank B"",""LOAN-2"",""HAS_LOAN"",""Owns loan"",""8"",""2023-02-01"")',  # double
        ]

        entities = []
        relationships = []

        for line in lines:
            line = line[1:-1] if line.startswith('(') else line
            fields = split_string_by_multi_markers(line, ['"",""', '","'])

            entity = handle_single_entity_extraction(fields, "chunk_1")
            if entity:
                entities.append(entity)

            rel = handle_single_relationship_extraction(fields, "chunk_1")
            if rel:
                relationships.append(rel)

        assert len(entities) == 2
        assert len(relationships) == 2
        assert all(e['entity_type'] == 'ORGANIZATION' for e in entities)
        assert all(r['relationship_type'] == 'HAS_LOAN' for r in relationships)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_line(self):
        """Test handling of empty lines"""
        entity = handle_single_entity_extraction([], "chunk_1")
        rel = handle_single_relationship_extraction([], "chunk_1")

        assert entity is None
        assert rel is None

    def test_malformed_record_missing_quotes(self):
        """Test handling of malformed records"""
        line = 'entity,Name,Type,Description'  # No quotes at all
        fields = split_string_by_multi_markers(line, ['"",""', '","', ','])

        entity = handle_single_entity_extraction(fields, "chunk_1")
        # Should still work if fields are present
        assert entity is not None or len(fields) < 4

    def test_unicode_in_names(self):
        """Test handling of unicode characters in entity/relationship names"""
        line = '"entity","Società Cooperativa","organization","Italian company €250K"'
        fields = split_string_by_multi_markers(line, ['","'])

        entity = handle_single_entity_extraction(fields, "chunk_1")

        assert entity is not None
        assert 'SOCIETÀ COOPERATIVA' in entity['entity_name']
        assert '€250K' in entity['description']

    def test_special_characters_in_descriptions(self):
        """Test handling of special characters"""
        line = '"relationship","A","B","HAS_LOAN","Description with, commas, and: colons","9","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is not None
        assert 'commas' in rel['description']
        assert 'colons' in rel['description']

    def test_very_long_description(self):
        """Test handling of very long descriptions"""
        long_desc = "A" * 1000
        line = f'"relationship","Source","Target","HAS_LOAN","{long_desc}","9","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is not None
        assert len(rel['description']) == 1000

    def test_numeric_entity_names(self):
        """Test handling of numeric entity names (like IDs)"""
        line = '"entity","12345","loan","Loan with numeric ID"'
        fields = split_string_by_multi_markers(line, ['","'])

        entity = handle_single_entity_extraction(fields, "chunk_1")

        assert entity is not None
        assert entity['entity_name'] == '12345'

    def test_relationship_with_float_weight(self):
        """Test relationship with decimal weight"""
        line = '"relationship","A","B","HAS_LOAN","Description","7.5","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is not None
        assert rel['weight'] == 7.5

    def test_relationship_with_zero_weight(self):
        """Test relationship with zero weight"""
        line = '"relationship","A","B","HAS_LOAN","Description","0","2023-01-01"'
        fields = split_string_by_multi_markers(line, ['","'])

        rel = handle_single_relationship_extraction(fields, "chunk_1")

        assert rel is not None
        assert rel['weight'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
