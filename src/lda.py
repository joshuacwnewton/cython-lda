from interactive_plot import *
from numpy import argsort, cumsum, log, ones, random, searchsorted, sum, zeros
import os, sys, shutil

import cy_lda


class LDA(object):

    def log_prob(self):

        beta, beta_sum = self.beta, self.beta_sum
        alpha, alpha_sum = self.alpha, self.alpha_sum
        nwt, nt, ntd = self.nwt, self.nt, self.ntd

        lp = 0.0

        nwt.fill(0)
        nt.fill(0)

        ntd.fill(0)

        for d, (doc, zd) in enumerate(zip(self.corpus, self.z)):
            for n, (w, t) in enumerate(zip(doc.tokens, zd)):

                lp += log((nwt[w, t] + beta[w]) / (nt[t] + beta_sum) * (ntd[t, d] + alpha[t]) / (n + alpha_sum))

                nwt[w, t] += 1
                nt[t] += 1
                ntd[t, d] += 1

        return lp

    def print_topics(self, num=5):

        beta = self.beta
        nwt = self.nwt

        alphabet = self.corpus.alphabet

        for t in range(self.T):

            sorted_types = list(map(alphabet.lookup, argsort(nwt[:, t] + beta)))
            print('Topic %s: %s' % (t+1, ' '.join(sorted_types[-num:][::-1])))

    def save_state(self, filename):

        alphabet = self.corpus.alphabet

        f = open(filename, 'w')

        for d, (doc, zd) in enumerate(zip(self.corpus, self.z)):
            for n, (w, t) in enumerate(zip(doc.tokens, zd)):
                f.write('%s %s %s %s %s\n' % (d, n, w, alphabet.lookup(w), t))

        f.close()

    def __init__(self, corpus, T, S, optimize, dirname):

#        random.seed(1000)

        self.corpus = corpus

        self.D = D = len(corpus)
        self.N = N = sum(len(doc) for doc in corpus)
        self.W = W = len(corpus.alphabet)

        self.T = T
        self.S = S

        self.optimize = optimize

        self.dirname = dirname

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        else:
            print("Clearing the '{}' directory".format(dirname))
            shutil.rmtree(dirname)
            os.makedirs(dirname)

        assert not os.listdir(dirname), 'Output directory must be empty.'

        print('# documents =', D)
        print('# tokens =', N)
        print('# unique types =', W)
        print('# topics =', T)
        print('# iterations =', S)
        print('Optimize hyperparameters =', optimize)

        self.beta = 0.01 * ones(W)
        self.beta_sum = 0.01 * W

        self.alpha = 0.1 * ones(T)
        self.alpha_sum = 0.1 * T

        self.nwt = zeros((W, T), dtype=int)
        self.nt = zeros(T, dtype=int)

        self.ntd = zeros((T, D), dtype=int)

        self.z = z = []

        for doc in corpus:
            z.append(zeros(len(doc), dtype=int))

    def inference(self):

        self.sample_topics(init=True)

        lp = self.log_prob()

        plt = InteractivePlot('Iteration', 'Log Probability')
        plt.update_plot(0, lp)

        print('\nIteration %s: %s' % (0, lp))
        self.print_topics()

        for s in range(1, self.S+1):

            sys.stdout.write('.')

            if not(s % 10):

                lp = self.log_prob()

                plt.update_plot(s, lp)

                print('\nIteration %s: %s' % (s, lp))
                self.print_topics()

                self.save_state('%s/state.txt.%s' % (self.dirname, s))

            self.sample_topics()

    def sample_topics(self, init=False):

        beta, beta_sum = self.beta, self.beta_sum
        alpha, alpha_sum = self.alpha, self.alpha_sum
        nwt, nt, ntd = self.nwt, self.nt, self.ntd

        for d, (doc, zd) in enumerate(zip(self.corpus, self.z)):
            for n, (w, t) in enumerate(zip(doc.tokens, zd)):
                if not init:
                    nwt[w, t] -= 1
                    nt[t] -= 1
                    ntd[t, d] -= 1

                dist = (nwt[w, :] + beta[w]) / (nt + beta_sum) * (ntd[:, d] + alpha)

                dist_sum = cumsum(dist)
                r = random.random() * dist_sum[-1]
                t = searchsorted(dist_sum, r)

                nwt[w, t] += 1
                nt[t] += 1
                ntd[t, d] += 1

                zd[n] = t

    def cy_inference(self):
        self.z = cy_lda.inference(
            S=self.S,
            T=self.T,
            corpus=self.corpus,
            z=self.z,
            nwt=self.nwt,
            nt=self.nt,
            ntd=self.ntd,
            alpha=self.alpha,
            alpha_sum=self.alpha_sum,
            beta=self.beta,
            beta_sum=self.beta_sum,
            dirname=self.dirname
        )

# Mistakes created:
# - Line 74: Clear out the directory if it exists
#   - When a directory already exists, cleaning it out is generally good practice as to prevent file overwriting issues
#   - This can be accomplished with a package like 'shutil'
