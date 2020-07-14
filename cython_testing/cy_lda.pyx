# cython: profile=True

"""
Build using `python cy_setup.py build_ext --inplace`,
then copy the generated `build/`, `.c`, and `.so` to `src/`.
"""

# from interactive_plot import *
from numpy import argsort, cumsum, log, random, searchsorted, pad, row_stack
# import sys


def print_topics(alphabet, beta, T, nwt, num=5):
    for t in range(T):
        sorted_types = list(map(alphabet.lookup, argsort(nwt[:, t] + beta)))
        print('Topic %s: %s' % (t+1, ' '.join(sorted_types[-num:][::-1])))


def save_state(corpus_arr, alphabet, z, filename):
    f = open(filename, 'w')
    for d, (doc, zd) in enumerate(zip(corpus_arr, z)):
        for n, (w, t) in enumerate(zip(doc, zd)):
            f.write('%s %s %s %s %s\n' % (d, n, w, alphabet.lookup(w), t))
    f.close()


def inference(S, T, corpus, z, nwt, nt, ntd, alpha, alpha_sum, beta, beta_sum,
              dirname, random_seed):
    def log_prob():
        lp = 0.0
        nwt.fill(0)
        nt.fill(0)
        ntd.fill(0)

        for d, (doc, zd) in enumerate(zip(corpus_arr, z)):
            for n, (w, t) in enumerate(zip(doc, zd)):
                lp += log(
                    (nwt[w, t] + beta[w]) / (nt[t] + beta_sum) *
                    (ntd[t, d] + alpha[t]) / (n + alpha_sum)
                )
                nwt[w, t] += 1
                nt[t] += 1
                ntd[t, d] += 1

        return lp

    def sample_topics(init=False):
        for d, (doc, zd) in enumerate(zip(corpus_arr, z)):
            for n, (w, t) in enumerate(zip(doc, zd)):
                if not init:
                    nwt[w, t] -= 1
                    nt[t] -= 1
                    ntd[t, d] -= 1

                dist = ((nwt[w, :] + beta[w]) / (nt + beta_sum) *
                        (ntd[:, d] + alpha))

                dist_sum = cumsum(dist)
                r = random.random() * dist_sum[-1]
                t = searchsorted(dist_sum, r)

                nwt[w, t] += 1
                nt[t] += 1
                ntd[t, d] += 1

                zd[n] = t

    random.seed(random_seed)

    # Decompose corpus class so documents can use a fixed-size array
    alphabet = corpus.alphabet
    tokens = [doc.tokens for doc in corpus]
    l = len(max(tokens, key=len))
    padded_tokens = [pad(t, (0, l-len(t)), constant_values=-1) for t in tokens]
    corpus_arr = row_stack(padded_tokens)

    sample_topics(init=True)
    lp = log_prob()
    # plt = InteractivePlot('Iteration', 'Log Probability')
    # plt.update_plot(0, lp)
    print('Iteration %s: %s' % (0, lp))
    # print_topics(corpus, beta, T, nwt)

    for s in range(1, S+1):
        # sys.stdout.write('.')
        if not(s % (S//10)):
            lp = log_prob()
            # plt.update_plot(s, lp)
            print('Iteration %s: %s' % (s, lp))
            # print_topics(corpus, beta, T, nwt)
            save_state(corpus_arr, alphabet, z,
                       '%s/state.txt.%s' % (dirname, s))
        sample_topics()