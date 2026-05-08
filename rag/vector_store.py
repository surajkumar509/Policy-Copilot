
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim=None):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim) if dim is not None else None
        self.items = []

    def add(self, vectors, texts):
        arr = np.array(vectors).astype('float32')
        if self.index is None:
            self.dim = arr.shape[1]
            self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(arr)
        self.items.extend(texts)

    def search(self, qvec, k=3):
        if self.index is None or self.index.ntotal == 0 or len(self.items) == 0:
            return []
        _, I = self.index.search(np.array([qvec]).astype('float32'), k)
        results = []
        for i in I[0]:
            if 0 <= i < len(self.items):
                results.append(self.items[i])
        return results

