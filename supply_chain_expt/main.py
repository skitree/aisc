import torch as ch
import numpy as np
import argparse
from torch import nn
from tqdm import tqdm
from models import Net, train_model
from explainer import explain, check_recourse_distance
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def get_trained_models(args):
    ch.manual_seed(0)
    X = ch.randn(1000, args.num_feats)
    ws = []
    ys = []
    for i in range(args.layers):
        ch.manual_seed(i)
        w = ch.rand(args.num_feats, 1)
        ws.append(w)
        ys.append(X @ w + ch.randn(1000, 1))
    # ws = [ch.rand(args.num_feats, 1) for _ in range(args.layers)]
    # ys = [X @ w + ch.randn(1000, 1) for w in ws]

    models = [Net(args.num_feats, 20, 1) if args.model == 'nn' else nn.Linear(args.num_feats, 1) for _ in range(args.layers)]
    models = [train_model(X, y, model) for y, model in zip(ys, models)]
    return models

def main(args):
    models = get_trained_models(args)

    # Create a test set
    ch.manual_seed(1)
    X_test = ch.randn(1000, args.num_feats)

    # Coarse explanation
    def make_predictor(model):
        def predictor(_X):
            y = model(ch.tensor(_X).float()).detach().numpy()
            return y
        return predictor

    last_Y = None
    final_exp = None
    for model in models:
        this_X = X_test.numpy() if last_Y is None else ch.cat([last_Y, X_test[:, :-1]], dim=1).numpy()
        exp = explain(make_predictor(model), this_X)
        if final_exp is None:
            final_exp = exp
        else:
            final_exp *= exp[:,0].unsqueeze(-1)
            final_exp[:,:-1] += exp[:,1:]

    # True explanation
    def true_predictor(_X):
        orig_X = ch.tensor(_X).clone()
        _X = ch.tensor(_X)
        for model in models:
            y = model(_X.float()).detach().numpy()
            _X = ch.cat([ch.tensor(y), orig_X[:,:-1]], dim=1)

        return y

    true_explanation = explain(true_predictor, X_test.numpy())
    true_recourse_distances = check_recourse_distance(true_explanation, true_predictor, X_test.clone())
    est_recourse_distances = check_recourse_distance(final_exp, true_predictor, X_test.clone())
    # Filter out the ones that are zero
    true_recourse_distances = true_recourse_distances[true_recourse_distances > 0]
    est_recourse_distances = est_recourse_distances[est_recourse_distances > 0]
    # Print out what fraction are under 100
    print(f"Fraction of true recourse distances under 100: {(true_recourse_distances < 100).mean():.3f}")
    print(f"Fraction of estimated recourse distances under 100: {(est_recourse_distances < 100).mean():.3f}")
    # Filter out the ones that are over 100
    filter = (true_recourse_distances < 100) & (est_recourse_distances < 100)
    true_recourse_distances = true_recourse_distances[filter]
    est_recourse_distances = est_recourse_distances[filter]
    # Print min, max, and mean recourse distances
    print(f"Mean true recourse distance: {true_recourse_distances.mean():.3f}")
    print(f"Mean estimated recourse distance: {est_recourse_distances.mean():.3f}")

    # Plot a histogram of the cosine similarity between the true and coarse explanations
    cos_sim = ch.nn.functional.cosine_similarity(final_exp, true_explanation, dim=1)
    print(f"Mean cosine similarity: {cos_sim.mean().item():.3f}")
    plt.hist(cos_sim.numpy(), bins=20, alpha=0.5, label=f'{args.layers} layers')
    # plt.xlim(0.8, 1)
    # plt.xlabel("Cosine similarity")
    # plt.ylabel("Frequency")
    # plt.show()
    return cos_sim 

if __name__ == '__main__':
    # Parse arguments: one for whether to use linear model or neural network, one for whether to use coarse or true explanation
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--explanation', type=str, default='coarse')
    parser.add_argument('--num-feats', type=int, default=10)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--tree-width', type=int, default=1)
    args = parser.parse_args()

    # main(args)

    cos_sims = {}
    results = []
    for l in [1, 2, 3, 4, 5]:
        args.layers = l
        results.append(main(args))
        cos_sims[l] = np.percentile(results[-1].numpy(), [2.5, 97.5])
    
    ch.save(cos_sims, 'cos_sims.pt')
    ch.save(results, 'results.pt')
    plt.show()
    

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, (lower, upper) in cos_sims.items():
        plt.plot([i, i], [lower, upper], marker='o')

    # Customizing the plot
    plt.xticks(list(cos_sims.keys()), [f'{i} Layers' for i in cos_sims])
    plt.title('5th to 95th Percentile Range for Each Array')
    plt.xlabel('Array')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()