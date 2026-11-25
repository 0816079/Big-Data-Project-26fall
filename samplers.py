import numpy as np

def entropy(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return -(p*np.log2(p) + (1-p) * np.log2(1-p))

class Random:
    def __init__(self, seed, **_):
        self.rng = np.random.default_rng(seed)

    def select(self, n, unlabeled, model, texts, it):
        idx = self.rng.choice(np.flatnonzero(unlabeled), min(n, unlabeled.sum()), replace=False)    
        return idx, -1, np.ones(len(idx), bool)
 
    def update(self, arm, reward, arm_yield):
        return False

class LTS:
    def __init__(self, seed, clusters, filter_label= True,  **_):
        self.rng, self.clusters, self.filter_label = np.random.default_rng(seed), clusters, filter_label
        k = clusters.max()+1
        self.a0, self.b0 = np.full(k,0.5),np.full(k,0.5)
        self.wins, self.losses, self.pulls = np.zeros(k), np.zeros(k), np.zeros(k, int)

    def arm(self, unlabeled):
        draw = self.rng.beta(self.wins + self.a0, self.losses + self.b0)
        draw[np.bincount(self.clusters[unlabeled], minlength=len(draw)) == 0] = -np.inf
        return int(draw.argmax())

    def select(self, n, unlabeled, model, texts, it):
        for _ in range(60):  
            arm = self.arm(unlabeled)
            members = np.flatnonzero((self.clusters == arm) & unlabeled)
            if not (self.filter_label and model):
                idx = self.rng.choice(members, min(n, len(members)), replace=False)
                break
            p = model.proba([texts[i] for i in members])
            pos, neg = members[p >= 0.5], members[p < 0.5]
            if len(pos):
                a = self.rng.choice(pos, min(n // 2, len(pos)), replace=False)
                idx = np.r_[a, self.rng.choice(neg, min(n - len(a), len(neg)), replace=False)]
                break
        else:
            idx = self.rng.choice(members, min(n, len(members)), replace=False)
        self.pulls[arm] += 1
        return idx.astype(int), arm, np.ones(len(idx), bool)    

    def _record(self, arm, win):
        self.wins[arm] += win
        self.losses[arm] += not win
        self.losses *= 0.99
        self.wins *= 0.99
        return win

    def update(self, arm, reward, arm_yield):
        return self._record(arm, reward >0)

class Hybrid(LTS):
    def __init__(self, seed, clusters, retrieval, prior=3.0, w_ret=0.5, w_ent=0.5, warmup=1,
                 quota=0.25, epsilon=0.1, reward_type="mixed", **_):
        super().__init__(seed, clusters, filter_label=False)
        self.ret, self.w_ret, self.w_ent, self.warmup = retrieval, w_ret, w_ent, warmup
        self.quota, self.epsilon, self.reward_type, self.yields = quota, epsilon, reward_type, []
        top = np.argsort(-retrieval)[: max(50, len(retrieval) // 20)]
        density = np.bincount(clusters[top], minlength=len(self.a0)) / np.bincount(clusters)
        self.prior_share = density / density.max()
        self.a0 = self.a0 + prior * self.prior_share
        self.b0 = self.b0 + prior * (1 - self.prior_share)

    def _pick(self, cand, n, model, texts, it):
        if len(cand) <= n:
            return cand
        if len(cand) > 3000:  
            cand = np.unique(np.r_[cand[np.argsort(-self.ret[cand])[:1500]], self.rng.choice(cand, 1500, replace=False)])
        score = self.w_ret * self.ret[cand]
        if self.w_ent and model and it >= self.warmup:
            score = score + self.w_ent * entropy(model.proba([texts[i] for i in cand]))
        order = cand[np.argsort(-score, kind="stable")]
        n_rand = round(self.epsilon * n)
        rest = order[n - n_rand:]
        return np.r_[order[: n - n_rand], self.rng.choice(rest, min(n_rand, len(rest)), replace=False)]

    def select(self, n, unlabeled, model, texts, it):
        arm = self.arm(unlabeled)
        a = self._pick(np.flatnonzero((self.clusters == arm) & unlabeled), n - round(self.quota * n), model, texts, it)
        left = unlabeled.copy()
        left[a] = False
        g = self._pick(np.flatnonzero(left), n - len(a), model, texts, it) if len(a) < n else np.array([], int)
        return np.r_[a, g].astype(int), arm, np.r_[np.ones(len(a), bool), np.zeros(len(g), bool)]

    def update(self, arm, reward, arm_yield):
        yield_win = arm_yield is not None and arm_yield > (np.mean(self.yields) if self.yields else 0)
        if arm_yield is not None:
            self.yields.append(arm_yield)
        win = {"metric": reward > 0, "yield": yield_win, "mixed": reward > 0 or yield_win}[self.reward_type]
        return self._record(arm, win)