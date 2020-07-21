import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation as SklearnLDA
from sklearn.feature_extraction.text import CountVectorizer

from cy_lda import CythonLDA
from py_lda import PythonLDA

import cProfile
import pstats
from pstats import SortKey


for j in range(5):
    raw_samples, _ = fetch_20newsgroups(
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        return_X_y=True
    )

    pattern = re.compile(r'(?u)\b\w\w+\b')
    samples = [pattern.findall(s.lower()) for s in raw_samples]

    with open("stopwords.txt", "r") as f:
        stopword_list = set(f.read().splitlines())

    # Parameters used across all tests
    t = 10
    beta = 0.01
    alpha = 0.1
    parameters = [(1000, 1000, 100),                     # Base case
                  (100, 1000, 100), (10000, 1000, 100),  # Vary N
                  (1000, 100, 100), (1000, 10000, 100),  # Vary L
                  (1000, 1000, 10), (1000, 1000, 1000)]  # Vary S
    mapping = {"1000": 100, "100": 10, "10": 5}
    for i, (n, w, s) in enumerate(parameters):
        print("--------------------------------------------------------------")
        print(f"       Starting iteration {i}... (N={n}, Wmax={w}, S={s})    ")
        print("--------------------------------------------------------------")
        fstr = f"{j}_{i}_n{n}_w{w}_s{mapping[f'{s}']}"

        # 1. Existing Scikit-learn implementation (Variational Bayes)
        tf_vectorizer = CountVectorizer(stop_words=stopword_list)
        vectors = tf_vectorizer.fit_transform(raw_samples)
        subset = vectors[:n]
        sk_lda = SklearnLDA(n_components=t, max_iter=mapping[f"{s}"],
                            topic_word_prior=beta, doc_topic_prior=alpha)
        cProfile.runctx('sk_lda.fit(subset)', globals(), locals(),
                        filename="log/"+fstr+"_sk.txt")
        pstats.Stats(
            "log/"+fstr+"_sk.txt"
        ).strip_dirs().sort_stats(SortKey.TIME).print_stats(10)

        # Prepare dataset for remaining LDA tests
        fstr = f"{j}_{i}_n{n}_w{w}_s{s}"
        subset = [[word for word in sample if word not in stopword_list]
                  for sample in samples if 0 < len(sample) < w]
        if len(subset) > n:
            subset = subset[:n]

        # 2. New Cython implementation (Collapsed Gibbs Sampling)
        cy_lda = CythonLDA(corpus=subset, T=t, S=s, beta=beta, alpha=alpha)
        cProfile.runctx('cy_lda.fit()', globals(), locals(),
                        filename="log/"+fstr+"_cy.txt")
        pstats.Stats(
            "log/"+fstr+"_cy.txt"
        ).strip_dirs().sort_stats(SortKey.TIME).print_stats(10)

        # 3. New NumPy implementation (Collapsed Gibbs Sampling)
        py_lda = PythonLDA(corpus=subset, T=t, S=s, beta=beta, alpha=alpha)
        cProfile.runctx('py_lda.fit()', globals(), locals(),
                        filename="log/"+fstr+"_py.txt")
        py_lda.print_topics()
        pstats.Stats(
            "log/"+fstr+"_py.txt"
        ).strip_dirs().sort_stats(SortKey.TIME).print_stats(10)


