from argparse import ArgumentParser
from corpus import *
from lda import *


def main():

    # parse command-line arguments

    p = ArgumentParser()

    p.add_argument('-input_file', metavar='input-file', help='file containing preprocessed data')
    p.add_argument('-output_dir', metavar='output-dir', default='../output', help='output directory')
    p.add_argument('-T', metavar='num-topics', type=int, default=100, help='number of topics (default: 100)')
    p.add_argument('-S', metavar='num-iterations', type=int, default=1000, help='number of Gibbs sampling iterations (default: 1000)')
    p.add_argument('--optimize', action='store_true', help='optimize Dirichlet hyperparameters')

    args = p.parse_args()

    print(' args  ', args)

    corpus = Corpus.load(args.input_file)

    lda = LDA(corpus, args.T, args.S, args.optimize, args.output_dir)

    lda.inference()


if __name__ == '__main__':

#    import cProfile
#    cProfile.run('main()', 'lda_profile')

    main()

# Mistake created:
# - Line 29: 'list()' instead of '[]'
#   - Slower and more obfuscated
# - Line 14 & 15: default values are not parameterized
#   - Should be parameterized to keep help message and default value consistent