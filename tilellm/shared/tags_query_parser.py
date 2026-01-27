import re
from typing import Dict, Any, List, Union, Optional


class TagQueryParser:
    """
    Parser per query booleane su tag con supporto NOT (!).

    Grammatica:
    expr     := or_expr
    or_expr  := and_expr ('|' and_expr)*
    and_expr := not_expr ('&' not_expr)*
    not_expr := '!' not_expr | primary   <-- NOT ha precedenza massima
    primary  := tag | '(' expr ')'
    """

    def __init__(self, field_name: str = "tags"):
        self.field = field_name
        self.tokens: List[str] = []
        self.pos: int = 0

    def parse(self, query: str) -> Dict[str, Any]:
        query = query.strip()
        if query.startswith(f"{self.field}:"):
            query = query[len(self.field) + 1:].strip()

        self.tokens = self._tokenize(query)
        self.pos = 0

        result = self._parse_or()

        if self.pos != len(self.tokens):
            raise ValueError(f"Token inaspettato: {self.tokens[self.pos]}")

        return result

    def _tokenize(self, query: str) -> List[str]:
        """Tokenizza: riconosce ! | & ( ) e tags"""
        # AGGIORNAMENTO: aggiunto ! nel primo gruppo
        pattern = r'(\(|\)|\||&|!)|([^\s()|&!]+)'
        tokens = []
        for op, tag in re.findall(pattern, query):
            if op:
                tokens.append(op)
            elif tag:
                tokens.append(tag)
        return tokens

    def _parse_or(self) -> Dict[str, Any]:
        left = self._parse_and()
        while self._match('|'):
            right = self._parse_and()
            left = {"$or": [left, right]}
        return left

    def _parse_and(self) -> Dict[str, Any]:
        left = self._parse_not()
        while self._match('&'):
            right = self._parse_not()
            left = {"$and": [left, right]}
        return left

    def _parse_not(self) -> Dict[str, Any]:
        """Gestisce la negazione ! (precedenza più alta)"""
        if self._match('!'):
            operand = self._parse_not()  # Ricorsivo per gestire !!a -> a
            return self._negate_filter(operand)
        return self._parse_primary()

    def _parse_primary(self) -> Dict[str, Any]:
        if self._match('('):
            expr = self._parse_or()
            if not self._match(')'):
                raise ValueError("Parentesi chiusa mancante")
            return expr

        return self._parse_tag()

    def _parse_tag(self) -> Dict[str, Any]:
        if self.pos >= len(self.tokens):
            raise ValueError("Tag atteso, fine query trovata")

        tag = self.tokens[self.pos]
        if tag in ('|', '&', '!', '(', ')'):
            raise ValueError(f"Atteso tag, trovato operatore: {tag}")

        self.pos += 1
        return {self.field: {"$in": [tag]}}

    def _negate_filter(self, filter_dict: Dict) -> Dict:
        """Applica la negazione logica ottimizzata per Pinecone"""

        # Doppia negazione: !!a -> a
        if "$not" in filter_dict:
            return filter_dict["$not"]

        # Tag semplice: {"tags": {"$in": ["a"]}} -> {"tags": {"$nin": ["a"]}}
        if self.field in filter_dict:
            field_cond = filter_dict[self.field]

            if "$in" in field_cond:
                return {self.field: {"$nin": field_cond["$in"]}}

            if "$nin" in field_cond:
                return {self.field: {"$in": field_cond["$nin"]}}

        # Espressione complessa: usa $not wrapper
        return {"$not": filter_dict}

    def _match(self, expected: str) -> bool:
        if self.pos < len(self.tokens) and self.tokens[self.pos] == expected:
            self.pos += 1
            return True
        return False


# Helper function per uso rapido
def parse_tag_query(query: str, field: str = "tags") -> Dict[str, Any]:
    return TagQueryParser(field).parse(query)


def build_tags_filter(tags: Union[str, List[str], None], field: str = "tags") -> Optional[Dict[str, Any]]:
    """
    Build Pinecone filter from tags parameter.
    - If tags is None: return None
    - If tags is a string: parse as boolean expression using parse_tag_query
    - If tags is a list: create AND condition for all tags
    """
    if tags is None:
        return None
    
    if isinstance(tags, str):
        return parse_tag_query(tags, field)
    
    if isinstance(tags, list):
        if not tags:
            return None
        # Create AND condition for all tags
        if len(tags) == 1:
            return {field: {"$in": [tags[0]]}}
        # Multiple tags: $and with each tag condition
        conditions = []
        for tag in tags:
            conditions.append({field: {"$in": [tag]}})
        return {"$and": conditions}
    
    raise ValueError(f"Unsupported tags type: {type(tags)}")