"""Unit tests for tag query parser."""

import pytest
from tilellm.shared.tags_query_parser import (
    parse_tag_query, 
    build_tags_filter,
    TagQueryParser
)


class TestTagQueryParser:
    """Test TagQueryParser class."""
    
    def test_single_tag(self):
        """Test parsing a single tag."""
        result = parse_tag_query("python")
        expected = {"tags": {"$in": ["python"]}}
        assert result == expected
    
    def test_or_operator(self):
        """Test OR operator."""
        result = parse_tag_query("python|api")
        expected = {
            "$or": [
                {"tags": {"$in": ["python"]}},
                {"tags": {"$in": ["api"]}}
            ]
        }
        assert result == expected
    
    def test_and_operator(self):
        """Test AND operator."""
        result = parse_tag_query("python&api")
        expected = {
            "$and": [
                {"tags": {"$in": ["python"]}},
                {"tags": {"$in": ["api"]}}
            ]
        }
        assert result == expected
    
    def test_not_operator(self):
        """Test NOT operator."""
        result = parse_tag_query("!legacy")
        expected = {"tags": {"$nin": ["legacy"]}}
        assert result == expected
    
    def test_double_not(self):
        """Test double negation."""
        result = parse_tag_query("!!python")
        expected = {"tags": {"$in": ["python"]}}
        assert result == expected
    
    def test_parentheses(self):
        """Test parentheses grouping."""
        result = parse_tag_query("(python|api)&!legacy")
        expected = {
            "$and": [
                {
                    "$or": [
                        {"tags": {"$in": ["python"]}},
                        {"tags": {"$in": ["api"]}}
                    ]
                },
                {"tags": {"$nin": ["legacy"]}}
            ]
        }
        assert result == expected
    
    def test_complex_expression(self):
        """Test complex expression with multiple operators."""
        result = parse_tag_query("(python|javascript)&(api|rest)&!legacy")
        assert "$and" in result
        assert len(result["$and"]) == 2  # First: AND of two ORs, Second: NOT
        # Verify structure
        first_and = result["$and"][0]
        second_and = result["$and"][1]
        # First element should be AND of two ORs
        assert "$and" in first_and
        assert len(first_and["$and"]) == 2
        assert "$or" in first_and["$and"][0]
        assert "$or" in first_and["$and"][1]
        # Second element should be NOT condition
        assert "tags" in second_and
        assert "$nin" in second_and["tags"]
    
    def test_field_prefix(self):
        """Test query with field prefix."""
        result = parse_tag_query("tags:python")
        expected = {"tags": {"$in": ["python"]}}
        assert result == expected
    
    def test_invalid_syntax(self):
        """Test invalid syntax raises error."""
        parser = TagQueryParser()
        with pytest.raises(ValueError):
            parser.parse("python&")
        
        with pytest.raises(ValueError):
            parser.parse("(python")
        
        with pytest.raises(ValueError):
            parser.parse("python||api")
    
    def test_custom_field_name(self):
        """Test parser with custom field name."""
        parser = TagQueryParser(field_name="categories")
        result = parser.parse("python")
        expected = {"categories": {"$in": ["python"]}}
        assert result == expected


class TestBuildTagsFilter:
    """Test build_tags_filter function."""
    
    def test_none_input(self):
        """Test None input returns None."""
        result = build_tags_filter(None)
        assert result is None
    
    def test_empty_list(self):
        """Test empty list returns None."""
        result = build_tags_filter([])
        assert result is None
    
    def test_single_tag_list(self):
        """Test single tag in list."""
        result = build_tags_filter(["python"])
        expected = {"tags": {"$in": ["python"]}}
        assert result == expected
    
    def test_multiple_tags_list(self):
        """Test multiple tags in list (AND condition)."""
        result = build_tags_filter(["python", "api"])
        expected = {
            "$and": [
                {"tags": {"$in": ["python"]}},
                {"tags": {"$in": ["api"]}}
            ]
        }
        assert result == expected
    
    def test_string_input(self):
        """Test string input is parsed."""
        result = build_tags_filter("python|api")
        expected = {
            "$or": [
                {"tags": {"$in": ["python"]}},
                {"tags": {"$in": ["api"]}}
            ]
        }
        assert result == expected
    
    def test_unsupported_type(self):
        """Test unsupported type raises error."""
        with pytest.raises(ValueError):
            build_tags_filter(123)
    
    def test_custom_field(self):
        """Test custom field name."""
        result = build_tags_filter(["python"], field="categories")
        expected = {"categories": {"$in": ["python"]}}
        assert result == expected