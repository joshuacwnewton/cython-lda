# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: profile=True

from numpy import ones, array, pad, row_stack, zeros, empty, random

from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf

cdef extern from "math.h":
    double log(double x) nogil

cdef class CythonLDA:
    cdef long[:, ::1] corpus
    cdef long[:, ::1] z
    cdef Py_ssize_t D
    cdef Py_ssize_t N
    cdef Py_ssize_t W
    cdef Py_ssize_t T
    cdef Py_ssize_t S
    cdef double[:, ::1] nwt
    cdef double[::1] nt
    cdef double[:, ::1] ntd
    cdef double[::1] alpha_arr
    cdef double alpha_sum
    cdef double[::1] beta_arr
    cdef double beta_sum
    cdef double[:] dist
    cdef double[:] dist_sum
    
    def __init__(self, corpus, int T, int S, double beta, double alpha):
        self._init_corpus(corpus)
        
        self.D = self.corpus.shape[0]
        self.N = self.corpus.shape[1]
        self.T = T
        self.S = S

        self.beta_arr = beta * ones(self.W)
        self.beta_sum = beta * self.W

        self.alpha_arr = alpha * ones(self.T)
        self.alpha_sum = alpha * self.T

        self.nwt = zeros((self.W, self.T), dtype=float)
        self.nt = zeros(self.T, dtype=float)
        self.ntd = zeros((self.T, self.D), dtype=float)

        self.dist = empty(self.T)
        self.dist_sum = empty(self.T)
        
    def _init_corpus(self, corpus):
        word_map = {}
        for doc in corpus:
            for word in doc:
                if word not in word_map:
                    word_map[word] = len(word_map)
                word = word_map[word]
                
        corpus = [array([word_map[w] for w in d], dtype=int)
                  for d in corpus]
        z = [zeros(len(doc), dtype=int) for doc in corpus]
        
        l = len(max(corpus, key=len))
        padded_corpus = [pad(d, (0, l-len(d)), constant_values=-1)
                         for d in corpus]
        padded_z = [pad(zd, (0, l-len(zd)), constant_values=-1)
                    for zd in z]
        
        self.corpus = row_stack(padded_corpus)
        self.z = row_stack(padded_z)
        self.W = len(word_map)

    cdef double _log_prob(self):
        cdef double lp = 0.0
        cdef Py_ssize_t w, t, d, n
        
        self.nwt[:, :] = 0
        self.nt[:] = 0
        self.ntd[:, :] = 0

        for d in range(self.D):
            for n in range(self.N):
                w = self.corpus[d, n]
                t = self.z[d, n]

                # Dummy value used to pad corpus/z to fixed-length arrays
                if w is -1:
                    break

                lp += log(((self.nwt[w, t] + self.beta_arr[w]) /
                           (self.nt[t] + self.beta_sum)) *
                          ((self.ntd[t, d] + self.alpha_arr[t]) /
                           (n + self.alpha_sum)))

                self.nwt[w, t] += 1
                self.nt[t] += 1
                self.ntd[t, d] += 1

        return lp

    cpdef _sample_topics(self, bint init=False):
        cdef Py_ssize_t w, t, d, n
        cdef double r
        
        for d in range(self.D):
            for n in range(self.N):
                w = self.corpus[d, n]
                t = self.z[d, n]

                # Dummy value used to pad corpus/z to fixed-length arrays
                if w is -1:
                    break

                if not init:
                    self.nwt[w, t] -= 1
                    self.nt[t] -= 1
                    self.ntd[t, d] -= 1

                for t in range(self.T):
                    self.dist[t] = ((self.nwt[w, t] + self.beta_arr[w]) /
                                    (self.nt[t] + self.beta_sum) *
                                    (self.ntd[t, d] + self.alpha_arr[t]))

                    # Cython version of np.cumsum()
                    if t == 0:
                        self.dist_sum[t] = self.dist[t]
                    else:
                        self.dist_sum[t] = self.dist[t] + self.dist_sum[t - 1]


                # Use this to match Hanna's random state and validate
                # r = random.random() * self.dist_sum[T-1]
                # Much faster C random number generation
                r = (rand()/RAND_MAX) * self.dist_sum[self.T-1]

                # Cython version of np.searchsorted()
                if r <= self.dist_sum[0]:
                    t = 0
                else:
                    for t in range(1, self.T):
                        if self.dist_sum[t-1] < r <= self.dist_sum[t]:
                            break

                self.nwt[w, t] += 1
                self.nt[t] += 1
                self.ntd[t, d] += 1

                self.z[d, n] = t

    cdef inference_loop(self):
        cdef Py_ssize_t s
        cdef double lp

        self._sample_topics(init=True)
        lp = self._log_prob()
        printf('\nIteration %d: %f', 0, lp)

        for s in range(1, self.S+1):
            self._sample_topics()
            if not(s % (self.S//10)):
                lp = self._log_prob()
                printf('\nIteration %ld: %f', s, lp)

        printf('\n\n')

    cpdef fit(self):
        self.inference_loop()
