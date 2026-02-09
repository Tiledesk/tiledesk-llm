from pydantic import BaseModel, Field, SecretStr, model_validator
from typing import Optional, Literal


class Engine(BaseModel):
    name: str = Field(default="pinecone")
    type: Optional[str] = Field(default="serverless")
    apikey: Optional[SecretStr]| None = None
    vector_size: int = Field(default=1536)
    index_name: str = Field(default="tilellm")
    text_key: Optional[str] = Field(default="text")
    metric: str = Field(default="cosine")
    host: Optional[str] = Field(default="localhost")
    port: Optional[int] = Field(default=6333)
    deployment: Optional[Literal["local", "cloud"]] = Field(default="local")
    database: Optional[str] = Field(default="default")

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

        else:
            raise ValueError(f"Unsupported engine: {self.name}")

        return self

    @model_validator(mode='after')
    def set_text_key(self):
        if self.name == "pinecone":
            if self.type == "serverless":
                self.text_key = "text"
            elif self.type == "pod":
                self.text_key = "content"
        return self