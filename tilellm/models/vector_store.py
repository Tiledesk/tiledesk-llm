from pydantic import BaseModel, Field, SecretStr, model_validator
from typing import Optional, Literal


class Engine(BaseModel):
    name: str = Field(default="pinecone")
    type: Optional[str] = Field(default="serverless")
    apikey: Optional[SecretStr]| None = None
    vector_size: int = Field(default=1536)
    index_name: str = Field(default="tilellm")
    text_key: Optional[str] = Field(
        default="text",
        description=(
            "Metadata field that stores the chunk text. "
            "Defaults to 'text' on Pinecone serverless and 'content' on pod. "
            "Set it explicitly (e.g. 'content') when a serverless index was migrated "
            "from a pod and still stores the chunk under the pod field name."
        ),
    )
    metric: str = Field(default="cosine")
    host: Optional[str] = Field(default="localhost")
    port: Optional[int] = Field(default=6333)
    deployment: Optional[Literal["local", "cloud"]] = Field(default="local")
    database: Optional[str] = Field(default="default")
    cloud: str = Field(default="aws", description="Pinecone serverless cloud provider (aws, gcp, azure)")
    region: str = Field(default="eu-west-1", description="Pinecone serverless region (e.g. eu-west-1, us-east-1)")

    @model_validator(mode='after')
    def validate_fields(self):
        if self.name == "pinecone":
            if self.type is None:
                self.type = "serverless"
            if self.type not in ("serverless", "pod"):
                raise ValueError("Type must be 'serverless' or 'pod' for Pinecone")

        elif self.name == "qdrant":
            if not self.deployment:
                raise ValueError("Deployment is required for Qdrant (local/cloud)")

            self.type = self.deployment

            if self.deployment == "local" or self.deployment == "cloud":
                if not (self.host and self.port):
                    raise ValueError("Host and port are required for local Qdrant")

        elif self.name == "milvus":
            if not self.deployment:
                self.deployment = "local"
            if self.deployment not in ("local", "cloud"):
                raise ValueError("Deployment must be 'local' or 'cloud' for Milvus")
            
            self.type = self.deployment
            
            if not (self.host and self.port):
                raise ValueError("Host and port are required for Milvus")
            
            # Set default metric if not specified
            if not self.metric:
                self.metric = "COSINE"

        elif self.name == "redis":
            if "port" not in self.model_fields_set:
                self.port = 6379
            # Redis Search indexes only DB 0 ("Cannot create index on db != 0").
            if self.database in (None, "default"):
                self.database = "0"
            if str(self.database) != "0":
                raise ValueError("Redis vector store: database must be 0 (Redis Search indexes only DB 0)")
            self.type = None

        else:
            raise ValueError(f"Unsupported engine: {self.name}")

        return self

    @model_validator(mode='after')
    def set_text_key(self):
        """Default the chunk-text metadata field by index type, unless the caller set it.

        Pinecone's pod-to-serverless migration copies metadata as-is, so a migrated
        serverless index still stores the chunk under ``content``. Passing
        ``text_key="content"`` keeps that name. Omitting it still defaults to
        ``text`` (serverless) or ``content`` (pod).
        """
        if self.name != "pinecone":
            return self
        if "text_key" in self.model_fields_set and self.text_key:
            return self
        self.text_key = "content" if self.type == "pod" else "text"
        return self