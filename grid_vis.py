import argparse
import pickle
from itertools import product

from communityDetection import load_network
from models import *
from utils.util import eval


def main():
    scores = dict()
    grid = {
        'strategy': ['prob', 'freq'],
        'use_coreness': [True, False],
        'epsilon': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
    }

    for data_path, epsilon, strategy, use_coreness in product(*grid.values()):
        args = argparse.Namespace(
            networkFile=f'{data_path}.dat',
            communityFile=f'{data_path}-c.dat',
            communitySavePath=f'{data_path}.cmty',
        )
        graph = load_network(args.networkFile)
        labels = my_nlta(graph, epsilon, strategy, use_coreness, args.communityFile, max_epochs=20)
        score = eval(labels, args.communityFile)
    
        scores[f'{data_path.split('/')[-1]}-{epsilon}-{strategy}-{use_coreness}'] = score
    
    with open('./checkpoints/ours_with_coreness.pickle', 'wb') as f:
        pickle.dump(scores, f)
        
    return


if __name__ == '__main__':
    main()