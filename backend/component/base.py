#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")

# from modules.milvus_module import MilvusModule
# from modules.azure_openai_module import AzureOpenaiModule
# from utils import post_utils

from dataclasses import dataclass, fields, field
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import numpy as np
import uuid
import json
import textwrap


# @dataclass
# class ProjectInfo:
#     project_key: str
#     langcode: str = ""
#     project_langcode: str = ""
#     query_str: str = ""
#     rag_id: str = ""
#     sn: str = ""
#     platform: str = ""
#     # system_prompt: Optional[str] = None  # 允許手動傳入

#     def __post_init__(self):
#         self.langcode = post_utils.standardize_langCode(self.langcode)
#         self.project_langcode = post_utils.standardize_langCode(self.project_langcode)
#         if isinstance(self.query_str, (list, tuple)):
#             self.query_str = self.query_str[-1]
#         # if self.system_prompt is None:  # 如果沒傳，就根據 langcode 設定
#         #     self.system_prompt = prompter.system_prompt_templates[self.langcode]

# @dataclass
# class ExtendedProjectInfo(ProjectInfo):
#     additional_info: str = ""
#     # collection: str = ""

#     @classmethod
#     def from_args(cls, proj_info: ProjectInfo, additional_info: str = ""):
#         return cls(
#             project_key=proj_info.project_key,
#             # collection=f'rag_{proj_info.project_key}',
#             langcode=proj_info.langcode,
#             query_str=proj_info.query_str,
#             rag_id=proj_info.rag_id,
#             sn=proj_info.sn,
#             additional_info=additional_info
#         )
    
#         # return cls(**vars(proj_info), additional_info=additional_info)

# @dataclass
# class Connection:
#     azure_openai: AzureOpenaiModule
#     milvus_db: MilvusModule = None

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0

    def add_usages(self, usage: dict):
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.embedding_tokens += usage.get("embedding_tokens", 0)

    def get_usages(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens
        }
    
class FilterCondition(str, Enum):
    """Vector store filter conditions to combine different filters."""

    # TODO add more conditions
    AND = "and"
    OR = "or"
    NOT = "not"  # negates the filter condition

class FilterOperator(str, Enum):
    """Vector store filter operator."""

    # TODO add more operators
    EQ = "=="  # default operator (string, int, float)
    GT = ">"  # greater than (int, float)
    LT = "<"  # less than (int, float)
    NE = "!="  # not equal to (string, int, float)
    GTE = ">="  # greater than or equal to (int, float)
    LTE = "<="  # less than or equal to (int, float)
    IN = "in"  # In array (string or number)
    NIN = "nin"  # Not in array (string or number)
    ANY = "any"  # Contains any (array of strings)
    ALL = "all"  # Contains all (array of strings)
    TEXT_MATCH = "text_match"  # full text match (allows you to search for a specific substring, token or phrase within the text field)
    TEXT_MATCH_INSENSITIVE = (
        "text_match_insensitive"  # full text match (case insensitive)
    )
    CONTAINS = "contains"  # metadata array contains value (string or number)
    IS_EMPTY = "is_empty"  # the field is not exist or empty (null or empty array)

class MetadataFilter(BaseModel):
    key: str
    operator: FilterOperator
    value: Any



class MetadataFilters(BaseModel):
    """Metadata filters for vector stores."""

    # Exact match filters and Advanced filters with operators like >, <, >=, <=, !=, etc.
    filters: List[Union[MetadataFilter, "MetadataFilters"]]
    # and/or such conditions for combining different filters
    condition: Optional[FilterCondition] = FilterCondition.AND


class Node(BaseModel):
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Dict[int, float]] = None
    # filename: Optional[str] = None
    node_type: str = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.id_
    
    @property
    def node_id(self) -> str:
        return self.id_

    @node_id.setter
    def node_id(self, value: str) -> None:
        self.id_ = value

    @field_validator("sparse_embedding", mode="before")
    def convert_sparse_values(cls, v):
        if v is None:
            return v
        return {k: float(val) if isinstance(val, np.generic) else val for k, val in v.items()}

    def __str__(self) -> str:
        TRUNCATE_LENGTH = 350
        WRAP_WIDTH = 70
        source_text_truncated = self.truncate_text(
            self.text.strip(), TRUNCATE_LENGTH
        )
        source_text_wrapped = textwrap.fill(
            f"Text: {source_text_truncated}\n", width=WRAP_WIDTH
        )
        return f"Node ID: {self.node_id}\n{source_text_wrapped}"

    def get_filename(self) -> str:

        filename = self.metadata.get("filename", None)
        if filename==None:
            filename = self.metadata.get("file_name", "")
        
        return filename
    
    def truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to a maximum length."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."



# from concurrent.futures import ThreadPoolExecutor
# import time
# def func_1():
#     base_info = ProjectInfo("vivi_test", "en-us")
#     ext_project_info = ExtendedProjectInfo.from_args(base_info, additional_info="extra1")
#     time.sleep(2)
#     print(f'func_1: {ext_project_info.additional_info}')

# def func_2():
#     base_info = ProjectInfo("vivi_test_2", "zh-tw")
#     ext_project_info = ExtendedProjectInfo.from_args(base_info, additional_info="extra2")
#     print(f'func_2: {ext_project_info.additional_info}')

# if __name__ == "__main__":

#     # futures = []
#     # with ThreadPoolExecutor(max_workers=5) as executor:
#     #     futures.append(executor.submit(func_1))
#     #     futures.append(executor.submit(func_2))
    
#     # res = [future.result() for future in futures]

#     ### ======================================================================
#     request_data = {
#         "test": "hihi",
#         "project_key": "vivi_test_2",
#         "langcode": "zh-tw"
#     }
    
#     # proj_info = ProjectInfo(**filter_for_dataclass(request_data, ProjectInfo))

#     proj_info = ProjectInfo(project_key="vivi_test_2", langcode="zh-tw")
#     ext_project_info = ExtendedProjectInfo.from_args(proj_info, additional_info="extra")

#     print(ext_project_info.project_key)       # vivi_test
#     # print(ext_project_info.system_prompt)   # extra

#     ### ======================================================================

#     # base_info = ProjectInfo(project_key="vivi_test_2", langcode="zh-tw")
#     # conn = Connection(milvus_db=MilvusModule(), azure_openai=AzureOpenaiModule())

#     # infer_mgr = InferenceManager.from_args(conn=conn, base_info=base_info)

#     # res = infer_mgr.milvus_db.count_entities("rag_vivi_test")
#     # print(res)

#     # print(infer_mgr.project_key)

#     ### ======================================================================
#     doc = Node(text="123")
#     print(doc)