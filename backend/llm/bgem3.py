#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")

from typing import List
import time
from pathlib import Path


model_path = Path("./models/bge-m3")
# model_path = Path("/Users/viviliu/Documents/10_Vivi/ChatMyCV/backend/llm/bgem3.py")

# logger = DebugLog(module_name="BGEM3SparseEmbedding").logger

class BGEM3SparseEmbeddingFunction():
    def __init__(self, model_path) -> None:
        try:
            t = time.time()
            from FlagEmbedding import BGEM3FlagModel

            if model_path:
                self.model = BGEM3FlagModel(model_path, use_fp16=False)
            else:
                self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
            
            print(f"Successfully import BGEM3FlagModel, ({time.time()-t:.3f} sec)")
        except Exception as error_info:
            # error_info = (
            #     "Cannot import BGEM3FlagModel from FlagEmbedding. It seems it is not installed. "
            #     "Please install it using:\n"
            #     "pip install FlagEmbedding\n"
            # )
            # logger.fatal(error_info)
            print(error_info)
            sys.exit(1)

    def encode_queries(self, queries: List[str]):
        outputs = self.model.encode(
            queries, return_dense=False, return_sparse=True, return_colbert_vecs=False
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def encode_documents(self, documents: List[str]):
        outputs = self.model.encode(
            documents, return_dense=False, return_sparse=True, return_colbert_vecs=False
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result
    
bgem3_model = BGEM3SparseEmbeddingFunction(model_path=model_path)

if __name__ == "__main__":
    print(bgem3_model.encode_queries(["hello"]))