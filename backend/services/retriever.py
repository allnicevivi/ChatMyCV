# from pathlib import Path
# from typing import List, Optional

# from vectorstores.chroma.usage import (
#     get_collection_name_for_path,
#     get_persistent_client,
# )


# class ChromaRetriever:
#     def __init__(self, file_path: str | Path, persist_dir: Optional[str | Path] = None):
#         self.file_path = Path(file_path)
#         self.persist_dir = persist_dir
#         self.client = get_persistent_client(persist_dir)
#         # We don't know the full hash here; collections are named with hash suffix.
#         # At query-time, compute the current hash to target the right collection.

#     def _get_collection(self):
#         from vectorstores.chroma.usage import compute_file_sha256

#         current_hash = compute_file_sha256(self.file_path)
#         name = get_collection_name_for_path(self.file_path, current_hash)
#         return self.client.get_collection(name)

#     def query(self, query_text: str, k: int = 5):
#         collection = self._get_collection()
#         result = collection.query(query_texts=[query_text], n_results=k, include=["documents", "metadatas", "distances"])
#         docs: List[str] = result.get("documents", [[]])[0]
#         metas = result.get("metadatas", [[]])[0]
#         dists = result.get("distances", [[]])[0]
#         return list(zip(docs, metas, dists))


