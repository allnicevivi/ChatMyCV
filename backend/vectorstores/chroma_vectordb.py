#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

from backend.modules.azure_module import AzureModule
from utils.app_logger import LoggerSetup

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api.models.Collection import Collection

logger = LoggerSetup("ChromaUsage").logger

DEFAULT_CHROMA_DIR = Path(__file__).parent / ".chroma"

collection_name = "chat_cv"

class ChromaUsage:
    def __init__(
        self, 
        collection_name, 
        persist_dir: Optional[str | Path] = None,
        auto_create: bool = True
    ):

        self.persist_dir = Path(persist_dir) if persist_dir else DEFAULT_CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = collection_name

        self.collection = self.get_collection(collection_name)
        if not self.collection and auto_create:
            self.collection = self.create_collection(collection_name)

    # def get_collection_by_name(self, name: str) -> Optional[Collection]:
    #     try:
    #         return self.client.get_collection(name=name)
    #     except Exception:
    #         return None

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        # path = Path(file_path)
        # file_hash = self.compute_file_sha256(path)
        # collection_name = self.get_collection_name_for_path(path, file_hash)
        # return self.get_collection_by_name(collection_name)

        try:
            return self.client.get_collection(name=collection_name)
        except Exception:
            return None


    def create_collection(self, collection_name: str) -> Collection:
        
        # collection_name = collection_name if collection_name else file_path.name
        return self.client.create_collection(
            name=collection_name,
            # metadata={
            #     # "source": str(file_path.resolve()),
            #     # "hash": file_hash,
            # },
        )

    def get_data(self) -> dict[str, Any]:
        return self.collection.get()

    def get_existing_ids(self) -> List[str]:
        return [doc for doc in self.collection.get()["ids"]]

    def add_data_to_collection(
        self, 
        # file_path: str|Path,
        texts: List[str], 
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]=None, 
        node_id_prefix: str=collection_name
        # collection: Collection=None, 
        # collection_name: str="",
    ) -> None:

        # file_path = Path(file_path)

        # collection_name = collection_name if collection_name else collection.name
        # if not collection:
        #     collection = self.get_collection_by_name(collection_name)

        # if collection is None:
        #     raise ValueError(f"Collection '{collection_name}' does not exist.")
        # if metadatas is None:
        #     metadatas = [{"filename": file_path.name} for _ in texts]
        # else:
        #     metadatas = [{"filename": file_path.name, **m} for m in metadatas]


        existing_ids = self.get_existing_ids()
        ids_to_delete = [id for id in existing_ids if id.startswith(node_id_prefix)]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f'delete node_id_prefix="{node_id_prefix}", data: {len(ids_to_delete)}')

        ids = [f"{node_id_prefix}-{i}" for i in range(len(texts))]
        
        self.collection.add(documents=texts, metadatas=metadatas, embeddings=embeddings, ids=ids)
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
        k: int = 5
    ):

        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k, 
            include=["documents", "metadatas", "distances"]
        )

        docs: List[str] = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        return list(zip(docs, metas, dists))


    def list_all_collection_names(self) -> List[str]:
        """List all collection names in the client."""
        collections = self.client.list_collections()
        collection_names = [collection.name for collection in collections]
        return collection_names

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete a collection by name. Returns True if deletion succeeded."""
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
    import pprint as pp
    from pathlib import Path

    # Example file path and embedding provider
    file_path = Path("/Users/viviliu/Documents/10_Vivi/ChatMyCV/backend/data/en/resume_detail.md")
    azure_client = AzureModule()
    chroma_usage = ChromaUsage(collection_name="chat_cv_en")

    # Example texts, metadatas, embeddings
    texts = ["Sample text 1", "Sample text 2"]
    metadatas = [{"author": "Author 1"}, {"author": "Author 2"}]
    metadatas = [{"filename": file_path.name, **m} for m in metadatas]
    embeddings = azure_client.get_embeddings(texts)

    # # Ensuring or creating a collection
    # collection = chroma_usage.create_collection(
    #     file_path=file_path,
    # )

    # chroma_usage.add_data_to_collection(
    #     # file_path=Path("/Users/viviliu/Documents/10_Vivi/ChatMyCV/backend/data_2/resume.md"),
    #     texts=texts,
    #     metadatas=metadatas,
    #     embeddings=embeddings
    # )

    # print(chroma_usage.get_existing_ids())

    # # Querying a collection for a file
    # docs = chroma_usage.query_collection_for_file(
    #     collection_name=file_path.name,
    #     query_embedding=azure_client.get_embedding("sample"),
    # )
    # pp.pprint(docs)

    chroma_usage.delete_collection_for_file(filename="resume.md")

    # print(chroma_usage.get_existing_ids())
    # collection_names = chroma_usage.list_all_collection_names()
    # print(collection_names)

    # collection = chroma_usage.get_collection_by_name(file_path.name)
    # print(len(collection.get()))

    # result = chroma_usage.query_collection(
    #     azure_client.get_embeddings(["sample"]),
    # )
    # print(result)

    # result = chroma_usage.get_data()["metadatas"]
    # print(result)