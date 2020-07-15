# cython: profile=True

"""
Build using `python cy_setup.py build_ext --inplace`,
then copy the generated `build/`, `.c`, and `.so` to `src/`.
"""

import cython
from numpy import pad, row_stack, zeros, empty, int64
from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf

cdef extern from "math.h":
    double log(double x) nogil


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double log_prob(long[:, ::1] corpus, long[:, ::1] z,
                     double[:, ::1] nwt, double[::1] nt, double[:, ::1] ntd,
                     double[::1] alpha, double alpha_sum,
                     double[::1] beta, double beta_sum):
    cdef double lp = 0.0
    cdef double[:, ::1] nwt_copy = zeros((nwt.shape[0], nwt.shape[1]))
    cdef double[::1] nt_copy = zeros(nt.shape[0])
    cdef double[:, ::1] ntd_copy = zeros((ntd.shape[0], ntd.shape[1]))
    cdef Py_ssize_t w, t, d, n
    cdef Py_ssize_t D = corpus.shape[0]
    cdef Py_ssize_t N = corpus.shape[1]

    for d in range(D):
        for n in range(N):
            w = corpus[d, n]
            t = z[d, n]

            if w is -1:
                break

            lp += log(
                (nwt_copy[w, t] + beta[w]) / (nt_copy[t] + beta_sum) *
                (ntd_copy[t, d] + alpha[t]) / (n + alpha_sum)
            )

            nwt_copy[w, t] += 1
            nt_copy[t] += 1
            ntd_copy[t, d] += 1

    return lp


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef sample_topics(Py_ssize_t T, long[:, ::1] corpus, long[:, ::1] z,
                   double[:, ::1] nwt, double[::1] nt, double[:, ::1] ntd,
                   double[::1] alpha, double[::1] beta, double beta_sum,
                   bint init):
    cdef double[:] dist = empty(T)
    cdef double[:] dist_sum = empty(T)
    cdef Py_ssize_t t_idx = 0
    cdef Py_ssize_t w, t, d, n
    cdef Py_ssize_t D = corpus.shape[0]
    cdef Py_ssize_t N = corpus.shape[1]
    cdef double r

    for d in range(D):
        for n in range(N):
            w = corpus[d, n]
            t = z[d, n]

            # -1 are dummy values to ensure corpus/z are fixed-length arrays
            if w is -1:
                break

            if not init:
                nwt[w, t] -= 1
                nt[t] -= 1
                ntd[t, d] -= 1

            for t_idx in range(T):
                dist[t_idx] = ((nwt[w, t_idx] + beta[w]) /
                               (nt[t_idx] + beta_sum) *
                               (ntd[t_idx, d] + alpha[t_idx]))

                # Cython version of np.cumsum()
                if t_idx == 0:
                    dist_sum[t_idx] = dist[t_idx]
                else:
                    dist_sum[t_idx] = dist[t_idx] + dist_sum[t_idx - 1]


            # Use this to match Hanna's random state and validate
            # r = random.random() * dist_sum[T-1]
            # Much faster C random number generation
            r = (rand()/RAND_MAX) * dist_sum[T-1]

            # Cython version of np.searchsorted()
            if r <= dist_sum[0]:
                t = 0
            for t_idx in range(1, T):
                if dist_sum[t_idx-1] < r <= dist_sum[t_idx]:
                    t = t_idx
                    break

            nwt[w, t] += 1
            nt[t] += 1
            ntd[t, d] += 1

            z[d, n] = t


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inference_loop(Py_ssize_t S, Py_ssize_t T,
                    long[:, ::1] corpus, long[:, ::1] z,
                    double[:, ::1] nwt, double[::1] nt, double[:, ::1] ntd,
                    double[::1] alpha, double alpha_sum,
                    double[::1] beta, double beta_sum):
    cdef Py_ssize_t s
    cdef double lp

    sample_topics(T, corpus, z, nwt, nt, ntd, alpha, beta, beta_sum, True)
    lp = log_prob(corpus, z, nwt, nt, ntd, alpha, alpha_sum, beta, beta_sum)
    printf('\nIteration %d: %f', 0, lp)

    for s in range(1, S+1):
        if not(s % (S//10)):
            lp = log_prob(corpus, z, nwt, nt, ntd,
                          alpha, alpha_sum, beta, beta_sum)
            printf('\nIteration %ld: %f', s, lp)
        sample_topics(T, corpus, z, nwt, nt, ntd, alpha, beta, beta_sum, False)


def inference(py_S, py_T, py_corpus, py_z, py_nwt, py_nt, py_ntd,
              py_alpha, py_alpha_sum, py_beta, py_beta_sum, py_dirname):

    # Decompose corpus class
    py_alphabet = py_corpus.alphabet
    tokens = [doc.tokens for doc in py_corpus]

    # Convert lists of variable-length vectors into fixed-size arrays
    l = len(max(tokens, key=len))
    padded_tokens = [pad(t, (0, l-len(t)), constant_values=-1).astype(int64)
                     for t in tokens]
    py_corpus_arr = row_stack(padded_tokens)
    padded_z = [pad(zd, (0, l-len(zd)), constant_values=-1).astype(int64)
                for zd in py_z]
    py_z_arr = row_stack(padded_z)

    # Add typing to variables
    cdef Py_ssize_t S = py_S
    cdef Py_ssize_t T = py_T
    cdef long[:, ::1] corpus = py_corpus_arr
    cdef long[:, ::1] z = py_z_arr
    cdef double[:, ::1] nwt = py_nwt
    cdef double[::1] nt = py_nt
    cdef double[:, ::1] ntd = py_ntd
    cdef double[::1] alpha = py_alpha
    cdef double alpha_sum = py_alpha_sum
    cdef double[::1] beta = py_beta
    cdef double beta_sum = py_beta_sum

    inference_loop(S, T, corpus, z, nwt, nt, ntd,
                   alpha, alpha_sum, beta, beta_sum)

