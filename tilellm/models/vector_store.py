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

            if self.deployment == "local" or self.deployment == "cloud":
                if not (self.host and self.port):
                    raise ValueError("Host and port are required for local Qdrant")

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