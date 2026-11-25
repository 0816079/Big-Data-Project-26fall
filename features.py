import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Encoder:
    def __init__(self, texts, model="all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(model)
        except ImportError:
            print("[Encoder] sentence-transformers not installed -> TF-IDF + LSA fallback")
            self.st = None
            self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english").fit(texts)
            n_dim = min(256, len(self.tfidf.vocabulary_) - 1)
            self.svd = TruncatedSVD(n_dim, random_state=0).fit(self.tfidf.transform(texts))
        self.emb = self.encode(texts)
    def encode(self, texts):
        if self.st:
            return self.st.encode(texts, batch_size=64, normalize_embeddings=True)
        return normalize(self.svd.transform(self.tfidf.transform(texts)))


def lda_cluster(texts, k ,seed):
    counts = CountVectorizer(stop_words="english", min_df=2, max_df=0.95, max_features=20000).fit_transform(texts)
    topic = LatentDirichletAllocation(k, random_state=seed, max_iter=20).fit_transform(counts).argmax(1)
    return np.unique(topic, return_inverse=True)[1] 
    
def bm25(texts, query,k1 = 1.5, b =0.75):
    cv = CountVectorizer(stop_words="english")
    tf = cv.fit_transform(texts).tocsc().astype(float) 
    n_docs = tf.shape[0]
    doc_freq =np.diff(tf.indptr)
    idf = np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    doc_len = np.asarray(tf.sum(1)).ravel()
    norm = k1 * (1 - b + b * doc_len / doc_len.mean())
    scores = np.zeros(n_docs)
    for word in cv.transform([query]).indices: 
        col = tf.getcol(word)               
        rows, f = col.indices, col.data
        scores[rows] += idf[word] * f * (k1 + 1) / (f + norm[rows])
    return scores


def hybrid_scores(texts, encoder, query, rrf_k = 60):
    lexical = bm25(texts, query)
    dense = encoder.emb @ encoder.encode([query])[0]
    rank = lambda s: np.argsort(np.argsort(-s, kind="stable"))+1
    fused = 1 / (rrf_k + rank(lexical)) + 1 / (rrf_k + rank(dense))
    return (fused - fused.min()) / (fused.max() - fused.min() or 1)