from numpy import (argsort, ones, zeros, zeros_like,
                   cumsum, random, searchsorted, log)
from math import ceil


class PythonLDA:
    def __init__(self, corpus, T, S, beta, alpha):
        self._init_corpus(corpus)

        self.D = D = len(self.corpus)
        self.W = W = len(self.idx_to_word)
        self.T = T
        self.S = S

        self.beta_arr = beta * ones(W)
        self.beta_sum = beta * W

        self.alpha_arr = alpha * ones(T)
        self.alpha_sum = alpha * T

        self.nwt = zeros((W, T), dtype=float)
        self.nt = zeros(T, dtype=float)
        self.ntd = zeros((T, D), dtype=float)

        self.z = [zeros(len(doc), dtype=int) for doc in corpus]

    def _init_corpus(self, corpus):
        word_map = {}
        for doc in corpus:
            for word in doc:
                if word not in word_map:
                    word_map[word] = len(word_map)

        self.corpus = [[word_map[w] for w in d] for d in corpus]
        self.idx_to_word = {v: k for k, v in word_map.items()}

    def _log_prob(self):
        nwt = zeros_like(self.nwt)
        nt = zeros_like(self.nt)
        ntd = zeros_like(self.ntd)

        lp = 0.0
        for d, (doc, zd) in enumerate(zip(self.corpus, self.z)):
            for n, (w, t) in enumerate(zip(doc, zd)):
                numer_one = nwt[w, t] + self.beta_arr[w]
                denom_one = nt[t] + self.beta_sum
                numer_two = ntd[t, d] + self.alpha_arr[t]
                denom_two = n + self.alpha_sum
                per_word_val = log(numer_one/denom_one * numer_two/denom_two)
                lp += per_word_val

                nwt[w, t] += 1
                nt[t] += 1
                ntd[t, d] += 1

        return lp

    def _sample_topics(self, init=False):
        for d, (doc, zd) in enumerate(zip(self.corpus, self.z)):
            for n, (w, t) in enumerate(zip(doc, zd)):
                if not init:
                    self.nwt[w, t] -= 1
                    self.nt[t] -= 1
                    self.ntd[t, d] -= 1

                first_term = (self.nwt[w, :] + self.beta_arr[w]) / (
                            self.nt + self.beta_sum)
                second_term = (self.ntd[:, d] + self.alpha_arr)
                dist = first_term * second_term

                dist_sum = cumsum(dist)
                r = random.random() * dist_sum[-1]
                t = searchsorted(dist_sum, r)

                self.nwt[w, t] += 1
                self.nt[t] += 1
                self.ntd[t, d] += 1

                zd[n] = t

    def fit(self):
        self._sample_topics(init=True)
        lp = self._log_prob()
        print('Iteration %s: %s' % (0, lp))

        update_every = ceil(self.S / 10)
        for s in range(1, self.S + 1):
            self._sample_topics()
            if s % update_every == 0:
                lp = self._log_prob()
                print('Iteration %s: %s' % (s, lp))

        print()

    def print_topics(self, num=20):
        for t in range(self.T):
            highest_prob_words = argsort(self.nwt[:, t] + self.beta_arr)
            sorted_types = [self.idx_to_word[i] for i in highest_prob_words]
            print(
                'Topic %s: %s' % (t + 1, ' '.join(sorted_types[-num:][::-1])))