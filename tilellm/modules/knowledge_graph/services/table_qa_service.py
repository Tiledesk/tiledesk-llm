"""
Table QA Service
Generates and executes SQL queries on tables (Parquet) to answer analytical questions.
"""

import logging
import os
import duckdb
import pandas as pd
from typing import List, Dict, Any, Optional
from tilellm.shared.utility import get_service_config

logger = logging.getLogger(__name__)

class TableQAService:
    """
    Service to answer natural language questions about tables by translating them to SQL.
    Uses DuckDB to query Parquet files stored in MinIO.
    """

    def __init__(self, llm: Any):
        self.llm = llm
        self._init_duckdb()

    def _init_duckdb(self):
        """Initialize DuckDB with S3/MinIO configuration."""
        self.conn = duckdb.connect(database=':memory:')
        self.conn.execute("INSTALL httpfs; LOAD httpfs;")
        
        # Configure S3/MinIO credentials
        config = get_service_config()
        minio_conf = config.get("minio", {})
        
        minio_endpoint = minio_conf.get("endpoint", "localhost:9000")
        minio_access = minio_conf.get("access_key", "minioadmin")
        minio_secret = minio_conf.get("secret_key", "minioadmin")
        minio_secure = minio_conf.get("secure", False)
        
        # Remove 'http://' or 'https://' from endpoint if present for DuckDB config
        endpoint_host = minio_endpoint.replace("http://", "").replace("https://", "")
        
        use_ssl = "true" if minio_secure else "false"
        
        self.conn.execute(f"SET s3_endpoint='{endpoint_host}';")
        self.conn.execute(f"SET s3_access_key_id='{minio_access}';")
        self.conn.execute(f"SET s3_secret_access_key='{minio_secret}';")
        self.conn.execute(f"SET s3_use_ssl={use_ssl};")
        self.conn.execute("SET s3_url_style='path';") # MinIO often requires path style

    async def answer_query(
        self,
        query: str,
        tables: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Answer a query using the provided candidate tables.
        
        Args:
            query: User question
            tables: List of table dictionaries containing 'id', 'parquet_path', 'columns', 'description'
            
        Returns:
            Dict with 'answer', 'sql', 'data' (DataFrame result)
        """
        if not tables:
            return {"answer": None, "reason": "No tables provided"}

        # Get bucket name from config
        config = get_service_config()
        bucket_tables = config.get("minio", {}).get("bucket_tables", "document-tables")

        # 1. Select the most relevant table (simplification: take the first one or ask LLM)
        target_table = tables[0]
        
        # 2. Generate SQL
        parquet_url = f"s3://{bucket_tables}/{target_table.get('parquet_path')}"
        columns = target_table.get('columns', [])
        table_desc = target_table.get('description', '')
        
        sql_query = await self._generate_sql(query, columns, parquet_url, table_desc)
        
        if not sql_query:
             return {"answer": "Could not generate SQL query."}

        # 3. Execute SQL
        try:
            logger.info(f"Executing SQL: {sql_query}")
            # DuckDB execute returns a relation, fetchall gives list of tuples, fetch_df gives pandas
            result_df = self.conn.execute(sql_query).fetch_df()
            
            # 4. Synthesize Answer
            answer = await self._synthesize_answer(query, result_df, sql_query)
            
            return {
                "answer": answer,
                "sql": sql_query,
                "data": result_df.to_dict(orient='records'),
                "table_id": target_table.get('id')
            }
            
        except Exception as e:
            logger.error(f"SQL Execution failed: {e}")
            return {
                "answer": f"Failed to execute query on table data: {e}",
                "sql": sql_query
            }

    async def _generate_sql(self, query: str, columns: List[str], parquet_path: str, description: str) -> Optional[str]:
        """Generate DuckDB SQL query from natural language."""
        
        # Sanitize columns for prompt
        cols_str = ", ".join(columns)
        
        prompt = f"""You are a DuckDB SQL expert. Generate a SQL query to answer the user's question based on the table schema provided.

Table Description: {description}
Columns: {cols_str}
File Path: '{parquet_path}'

User Question: "{query}"

Rules:
1. Use the file path '{parquet_path}' in the FROM clause. Example: SELECT * FROM '{parquet_path}'
2. Use valid DuckDB syntax.
3. Return ONLY the SQL query, no markdown, no explanations.
4. If the question cannot be answered with the given columns, return "IMPOSSIBLE".
5. Be robust with column names (if they contain spaces, wrap in double quotes).
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            sql = response.content if hasattr(response, 'content') else str(response)
            sql = sql.strip().replace("```sql", "").replace("```", "")
            if "IMPOSSIBLE" in sql:
                return None
            return sql
        except Exception as e:
            logger.error(f"SQL Generation failed: {e}")
            return None

    async def _synthesize_answer(self, query: str, df: pd.DataFrame, sql: str) -> str:
        """Synthesize a natural language answer from the query result."""
        
        data_str = df.to_string(index=False, max_rows=10)
        
        prompt = f"""The user asked: "{query}"

I executed the following SQL:
{sql}

The result is:
{data_str}

Please interpret this result and provide a concise answer to the user's question.
"""
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Result: {data_str}"
