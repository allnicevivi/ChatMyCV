#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

from utils.app_logger import LoggerSetup

from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
import chromadb
from chromadb.api.models.Collection import Collection

logger = LoggerSetup("ChromaUsage").logger

DEFAULT_CHROMA_DIR = Path(__file__).parent / ".chroma"

collection_name = "chat_cv"


class ChromaUsage:
    def __init__(
        self,
        collection_name: str,
        persist_dir: Optional[str | Path] = None,
        auto_create: bool = True,
        distance_fn: Literal["cosine", "l2", "ip"] = "cosine"
    ):
        self.persist_dir = Path(persist_dir) if persist_dir else DEFAULT_CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = collection_name
        self.distance_fn = distance_fn

        self.collection = self.get_collection(collection_name)
        logger.info(f'Collection loaded: {self.collection}')
        if not self.collection and auto_create:
            self.collection = self.create_collection(collection_name)

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        try:
            return self.client.get_collection(name=collection_name)
        except Exception:
            return None

    def create_collection(self, collection_name: str) -> Collection:
        return self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.distance_fn}
        )

    def get_data(self) -> dict[str, Any]:
        return self.collection.get()

    def get_existing_ids(self) -> List[str]:
        return [doc for doc in self.collection.get()["ids"]]

    def add_data_to_collection(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]] = None,
        node_id_prefix: str = collection_name
    ) -> None:
        existing_ids = self.get_existing_ids()
        ids_to_delete = [id for id in existing_ids if id.startswith(node_id_prefix)]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f'delete node_id_prefix="{node_id_prefix}", data: {len(ids_to_delete)}')

        if metadatas is None:
            metadatas = [{} for _ in texts]

        ids = [f"{node_id_prefix}-{i}" for i in range(len(texts))]

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        logger.info(f'insert node_id_prefix="{node_id_prefix}", data: {len(ids)}')

    def delete_collection_for_file(self, file_path: str | Path=None, filename: str="") -> None:
        filename = file_path.name if not filename else filename
        try:
            existing_documents = self.collection.get()
            ids_to_delete = [
                doc_id for doc_id, metadata in zip(existing_documents["ids"], existing_documents["metadatas"])
                if metadata.get("filename") == filename
            ]
            self.collection.delete(ids=ids_to_delete)
            logger.info(f'delete file "{filename}", data: {len(ids_to_delete)}')
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}", exc_info=True)

    def query_collection(
        self,
        query_embedding: list[float],
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[tuple]:
        """
        Query the collection using dense embeddings.

        Args:
            query_embedding: Dense embedding vector for the query
            k: Number of results to return
            where: Optional metadata filter
            where_document: Optional full-text filter (e.g., {"$contains": "keyword"})

        Returns:
            List of tuples: (document, metadata, distance)
        """
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return [(doc, meta, dist) for doc, meta, dist in zip(docs, metas, dists)]

    def list_all_collection_names(self) -> List[str]:
        collections = self.client.list_collections()
        collection_names = [collection.name for collection in collections]
        return collection_names

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        target_name = collection_name or self.collection_name
        if not target_name:
            return False
        try:
            self.client.delete_collection(name=target_name)
            logger.info(f'delete collection "{target_name}"')
            self.collection = None
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {target_name}: {e}", exc_info=True)
            return False

if __name__ == "__main__":
    from pathlib import Path

    file_path = Path("./data/en/resume_detail.md")
    chroma_usage = ChromaUsage(collection_name="chat_cv_en", auto_create=False)

    # Example usage:
    # from backend.llm import bgem3
    #
    # texts = ["Sample text 1", "Sample text 2"]
    # metadatas = [{"author": "Author 1"}, {"author": "Author 2"}]
    # metadatas = [{"filename": file_path.name, **m} for m in metadatas]
    #
    # embeddings = bgem3.get_embeddings(texts)
    #
    # chroma_usage.add_data_to_collection(
    #     texts=texts,
    #     metadatas=metadatas,
    #     embeddings=embeddings.tolist(),
    # )
    #
    # query_embedding = bgem3.get_embeddings(["sample"], is_query=True)
    #
    # results = chroma_usage.query_collection(
    #     query_embedding=query_embedding[0].tolist(),
    # )
    # print(results)

    chroma_usage.delete_collection(collection_name="chat_cv_en")