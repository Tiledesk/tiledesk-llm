from pydantic import BaseModel, Field, model_validator
from typing import Optional, List


class ParametersScrapeType4(BaseModel):
    unwanted_tags: Optional[List[str]] = Field(default_factory=list)
    tags_to_extract: Optional[List[str]] = Field(default_factory=list)
    unwanted_classnames: Optional[List[str]] = Field(default_factory=list)
    desired_classnames: Optional[List[str]] = Field(default_factory=list)
    remove_lines: Optional[bool] = Field(default=True)
    remove_comments: Optional[bool] = Field(default=True)
    time_sleep: Optional[float] = Field(default=2)

    @model_validator(mode='after')
    def check_booleans(self):
        if self.remove_lines is None or self.remove_comments is None:
            raise ValueError('remove_lines and remove_comments must be provided in ParametersScrapeType4')
        return self