#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

from pathlib import Path
import asyncio

from component.base import Node
from llm import azure_client
from db.chroma_vectordb import ChromaUsage
from parsers import MarkdownReader
from utils.app_logger import LoggerSetup

logger = LoggerSetup("DocProcessor").logger

class DocProcessor:

    def __init__(self, lang: str) -> None:
        self.chroma_usage = ChromaUsage(collection_name=f"chat_cv_{lang}")

    
    def parse_doc(self, file_path: Path):
        return MarkdownReader().load_data(file=file_path)
    
    def store_doc(self, nodes: list[Node], file_path: Path):

        # Example texts, metadatas, embeddings
        texts = [n.text for n in nodes]
        metadatas = [n.metadata for n in nodes]
        dense_embeddings = asyncio.run(azure_client.embed(texts))

        # Ensuring or creating a collection
        node_cnt_before = len(self.chroma_usage.get_existing_ids())

        self.chroma_usage.add_data_to_collection(
            texts=texts,
            metadatas=metadatas,
            embeddings=dense_embeddings,
            node_id_prefix=file_path.name
        )

        node_cnt_after = len(self.chroma_usage.get_existing_ids())
        logger.info(f'node_cnt_before: {node_cnt_before}, node_cnt_after: {node_cnt_after}')


    def run(self, file_paths: list[str]|list[Path]):
        
        for file_path in file_paths:
            file_path = Path(file_path)
            nodes = self.parse_doc(file_path)
            self.store_doc(nodes, file_path)

        return nodes




# def load_or_build_collection_for_markdown(
#     file_path: str | Path,
#     embedder: BaseLLMProvider,
#     persist_dir: Optional[str | Path] = None,
#     chunk_size: int = 800,
#     chunk_overlap: int = 150,
# ):
#     """
#     Load an existing Chroma collection for the given markdown file if present;
#     otherwise parse, chunk, embed and persist a new collection.
#     """
#     path = Path(file_path)
#     reader = MarkdownReader()
#     nodes = reader.load_data(path)

#     texts: List[str] = []
#     metadatas: List[Dict[str, Any]] = []
#     for n in nodes:
#         if not n.text:
#             continue
#         texts.append(n.text)
#         meta = dict(n.metadata or {})
#         meta.setdefault("filename", path.name)
#         metadatas.append(meta)

#     # Embed only if a new collection must be created. ensure_collection will skip
#     # creation when a matching collection already exists.
#     # We optimistically attempt to create; if it already exists, embeddings are ignored.
#     embeddings: Optional[List[List[float]]] = None
#     if texts:
#         embeddings = embedder.get_embeddings(texts, dim=1536)

#     collection = ensure_collection(
#         file_path=path,
#         texts=texts if texts else None,
#         metadatas=metadatas if metadatas else None,
#         embeddings=embeddings if embeddings else None,
#         persist_dir=persist_dir,
#     )

#     return collection


if __name__ == "__main__":
    doc_processor = DocProcessor()


    file_paths = [
        Path("./data/resume.md"),
        Path("./data/resume_detail.md")
    ]
    # collection = load_or_build_collection_for_markdown(file_path, embedder=AzureModule())

    doc_processor.run(file_paths)


