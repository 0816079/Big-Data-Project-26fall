import numpy as np
from sklearn.neighbors import NearestNeighbors


def density_filter(emb, thresh=0.95, q =0.5, k =10):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(emb)), metric="cosine").fit(emb)
    dist, nbr = nn.kneighbors(emb)

    sim ,nbr = 1-dist[:,1:], nbr[:, 1:]
    density = sim.mean(1)
    dense = density >= np.quantile(density, q)

    dense_idx = np.flatnonzero(dense)
    rnn = NearestNeighbors(radius=1 - thresh, metric="cosine").fit(emb[dense_idx])
    groups = rnn.radius_neighbors(emb[dense_idx], return_distance=False)

    keep = np.ones(len(emb), bool)
    pos = {g: j for j, g in enumerate(dense_idx)}
    for i in sorted(dense_idx, key=lambda i: -density[i]):  
        if keep[i]:
            dup = dense_idx[groups[pos[i]]]
            keep[dup[dup != i]] = False
    return keep, density