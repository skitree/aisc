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

import warnings
warnings.filterwarnings("ignore")

def get_trained_models(args):
    seed = args.runs
    ch.manual_seed(seed)
    X = ch.randn(1000, args.num_feats)
    ws = []
    ys = []
    width = args.tree_width  # m children per node
    
    # Calculate nodes per layer in m-regular tree
    total_nodes = 0
    nodes_per_layer = []
    for i in range(args.layers):
        nodes_in_this_layer = width ** i
        nodes_per_layer.append(nodes_in_this_layer)
        total_nodes += nodes_in_this_layer
    
    node_count = 0
    for i in range(args.layers):
        for j in range(nodes_per_layer[i]):
            ch.manual_seed(seed * 1000 + node_count)
            w = ch.rand(args.num_feats, 1)
            ws.append(w)
            ys.append(X @ w + ch.randn(1000, 1))
            node_count += 1

    models = [Net(args.num_feats, 20, 1) if args.model == 'nn' 
              else nn.Linear(args.num_feats, 1) 
              for _ in range(total_nodes)]
    models = [train_model(X, y, model) for y, model in zip(ys, models)]
    
    return models

#use this if we change number of features at different depths
#I'm leaving this here with the asterisk that this changes our underlying explanations protocol thus I didn't end up using.
def get_trained_models_width_aware(args):
    ch.manual_seed(0)
    X = ch.randn(1000, args.num_feats)
    ws = []
    ys = []
    Xs = []  # Store the input data for each model
    width = args.tree_width
    
    # Calculate nodes per layer in m-regular tree
    total_nodes = 0
    nodes_per_layer = []
    for i in range(args.layers):
        nodes_in_this_layer = width ** (args.layers - 1 - i)  # Fixed formula for many-to-one tree
        nodes_per_layer.append(nodes_in_this_layer)
        total_nodes += nodes_in_this_layer
    
    node_count = 0
    for i in range(args.layers):
        for j in range(nodes_per_layer[i]):
            ch.manual_seed(node_count)
            if i == 0:
                input_size = args.num_feats
                w = ch.rand(input_size, 1)
                Xs.append(X)
                ys.append(X @ w + ch.randn(1000, 1))
            else:
                input_size = width + args.num_feats - 1
                w = ch.rand(input_size, 1)
                synthetic_X = ch.randn(1000, input_size)
                Xs.append(synthetic_X)
                ys.append(synthetic_X @ w + ch.randn(1000, 1))
            ws.append(w)
            node_count += 1

    # Create models with appropriate input sizes
    models = []
    for i in range(args.layers):
        for j in range(nodes_per_layer[i]):
            input_size = args.num_feats if i == 0 else width + args.num_feats-1
            if args.model == 'nn':
                models.append(Net(input_size, 20, 1))
            else:
                models.append(nn.Linear(input_size, 1))
    
    models = [train_model(X, y, model) for X, y, model in zip(Xs, ys, models)]
    
    return models

def main(args):
    if args.width_aware and args.tree_width > 1:
        models = get_trained_models_width_aware(args)
    else:
        args.width_aware = False
        models = get_trained_models(args)

    # Coarse explanation
    def make_predictor(model):
        def predictor(_X):
            y = model(ch.tensor(_X).float()).detach().numpy()
            return y
        return predictor

    def true_predictor(_X):
        orig_X = ch.tensor(_X).clone()
        _X = ch.tensor(_X)
        for model in models:
            y = model(_X.float()).detach().numpy()
            _X = ch.cat([ch.tensor(y), orig_X[:,:-1]], dim=1)

        return y
    
    def true_predictor_with_width(_X):
        orig_X = ch.tensor(_X).clone()
        _X = ch.tensor(_X)
        prev_layer_outputs = [_X]  # Store initial input
        node_idx = 0  # Track model index in 1D array

        for layer in range(args.layers):
            layer_size = args.tree_width ** (args.layers - layer - 1)  # Nodes in this layer
            current_outputs = []  # Store outputs for this layer
            
            if layer == 0:
                # First layer directly takes _X as input
                model_input = _X
                for node in range(layer_size):
                    y = models[node_idx](model_input.float()).detach().numpy()
                    current_outputs.append(ch.tensor(y))
                    node_idx += 1  # Move to next model
            else:
                for node in range(layer_size):  # Process nodes in this layer
                    # print(_X.shape)
                    parent_start_idx = node * args.tree_width  # First parent's index in 1D list
                    # selected_parents = [ch.tensor(p) if isinstance(p, np.ndarray) else p for p in prev_layer_outputs[parent_start_idx: parent_start_idx + args.tree_width]]
                    selected_parents = [ch.tensor(p[:, -1:]) if ch.tensor(p).dim() > 1 else ch.tensor(p).reshape(-1, 1) 
                                  for p in prev_layer_outputs[parent_start_idx: parent_start_idx + args.tree_width]]

                    # if args.width_aware:
                    if args.merge_inputs == 'cat':
                        concatenated_parents = ch.cat(selected_parents, dim=1)
                    elif args.merge_inputs == 'sum':
                        concatenated_parents = ch.stack(selected_parents, dim=1).sum(dim=1, keepdim=True).squeeze(dim=1)
                    elif args.merge_inputs == 'mean':
                        concatenated_parents = ch.stack(selected_parents, dim=1).mean(dim=1, keepdim=True).squeeze(dim=1)
                    #if we change number of features at different depths based on width, just concatenate
                    
                    # Append original features to maintain structure
                    model_input = ch.cat([concatenated_parents, orig_X[:, :-1]], dim=1)
                    # print(model_input.shape)
                    # Process the current node's model
                    y = models[node_idx](model_input.float()).detach().numpy()
                    current_outputs.append(y)
                    node_idx += 1  # Move to the next model

            prev_layer_outputs = current_outputs  # Store for the next layer

        return prev_layer_outputs[0]
    
    ch.manual_seed(1)
    X_test = ch.randn(1000, args.num_feats)

    last_Y = None
    final_exp = None

    for i, model in enumerate(models):
        print(args.width_aware, args.tree_width, i)
        # if args.width_aware and args.tree_width > 1 and i >= args.tree_width:
        #     # For non-root nodes in width-aware tree, extend input dimension
        #     synthetic_dims = ch.randn(1000, args.tree_width - 1)  # Additional parent dimensions
        #     this_X = ch.cat([X_test, synthetic_dims], dim=1).numpy()
        #     print(this_X.shape)
        if args.merge_inputs == 'cat' and args.tree_width > 1:

            nodes_per_layer = []
            layer_start_idx = [0]  # Starting index for each layer
            total_nodes = 0
            for i in range(args.layers):
                nodes_in_this_layer = args.tree_width ** (args.layers - 1 - i)
                nodes_per_layer.append(nodes_in_this_layer)
                total_nodes += nodes_in_this_layer
                layer_start_idx.append(total_nodes) 

            for i, model in enumerate(models):
                layer = next(idx for idx, start in enumerate(layer_start_idx[1:]) if i < start)
                this_X = X_test.numpy() if layer == 0 or last_Y is None else ch.cat([last_Y, X_test[:, :-args.tree_width-1]], dim=1).numpy()
        else:
            # For root nodes or non-width-aware
            this_X = X_test.numpy() if last_Y is None else ch.cat([last_Y, X_test[:, :-1]], dim=1).numpy()

        exp = explain(make_predictor(model), this_X)
        if final_exp is None:
            final_exp = exp
        else:
            final_exp *= exp[:,0].unsqueeze(-1)
            final_exp[:,:-1] += exp[:,1:]

    # True explanation
    # Return final model's output
    if args.tree_width == 1:
        true_explanation = explain(true_predictor, X_test.numpy())
        true_recourse_distances = check_recourse_distance(true_explanation, true_predictor, X_test.clone())
        est_recourse_distances = check_recourse_distance(final_exp, true_predictor, X_test.clone())
        # Filter out the ones that are zero
        true_recourse_distances = true_recourse_distances[true_recourse_distances > 0]
        est_recourse_distances = est_recourse_distances[est_recourse_distances > 0]

    else:
        true_explanation = explain(true_predictor_with_width, X_test.numpy())
        true_recourse_distances = check_recourse_distance(true_explanation, true_predictor_with_width, X_test.clone())
        est_recourse_distances = check_recourse_distance(final_exp, true_predictor_with_width, X_test.clone())
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

    print(f"Mean true recourse distance: {true_recourse_distances.mean():.3f}")
    print(f"Mean estimated recourse distance: {est_recourse_distances.mean():.3f}")

    cos_sim = ch.nn.functional.cosine_similarity(final_exp, true_explanation, dim=1)
    mse = ch.nn.functional.mse_loss(final_exp, true_explanation, reduction='none')
    l2_dist = ch.norm(final_exp - true_explanation, dim=1)

    print(f"Mean cosine: {cos_sim.mean().item():.3f}")
    print(f"Mean MSE: {mse.mean().item():.3f}")
    print(f"Mean L2 distance: {l2_dist.mean().item():.3f}")

    metrics = {
            'depth': args.layers,
            'width': args.tree_width,
            'merge_type': args.merge_inputs,
            'width_aware': args.width_aware,
            'true_recourse_under_100': (true_recourse_distances < 100).mean(),
            'est_recourse_under_100': (est_recourse_distances < 100).mean(),
            'mean_true_recourse': true_recourse_distances.mean(),
            'mean_est_recourse': est_recourse_distances.mean(),
            'mean_cosine_similarity': cos_sim.mean(),
            'mean_mse': mse.mean(),
            'mean_l2_dist': l2_dist.mean()
        }

    filename_base = f'd{args.layers}_w{args.tree_width}_m{args.merge_inputs}'
    if args.width_aware:
        filename_base += '_width_aware'
    filename_base += f'_run{args.runs}'

    ch.save(metrics, f'metrics_{filename_base}.pt')


    plt.hist(cos_sim.numpy(), bins=20, alpha=0.5, label=f'{args.layers} layers')
    # plt.xlim(0.8, 1)
    # plt.xlabel("Cosine similarity")
    # plt.ylabel("Frequency")
    # plt.show()



    return cos_sim, mse, l2_dist

if __name__ == '__main__':
    # Parse arguments: one for whether to use linear model or neural network, one for whether to use coarse or true explanation
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--explanation', type=str, default='coarse')
    parser.add_argument('--num-feats', type=int, default=10)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--tree-width', type=int, default=1)
    parser.add_argument('--merge-inputs', type=str, default='sum')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--width-aware', action='store_true', help="Enable width-aware model")
    args = parser.parse_args()

    # main(args)

    # First experiment: Depth only
    # cos_sims = {}
    # results = []
    # for l in [1, 2, 3, 4, 5]:
    #     args.layers = l
    #     result = main(args)
    #     results.append(result)
    #     cos_sims[l] = np.percentile(result.numpy(), [2.5, 97.5])
    #     # Save individual results with depth in filename
    #     ch.save(result, f'result_d{l}_w1.pt')
    #     ch.save(cos_sims[l], f'cos_sims_d{l}_w1.pt')
    
    # # Save the full dictionary for easier plotting
    # ch.save(cos_sims, 'cos_sims_depth_all.pt')
    # plt.show()

    exp_id = f'{args.merge_inputs}_total_width{args.tree_width}_total_depth{args.layers}{"_width_aware" if args.width_aware else ""}'
    
    all_results = {}
    runs = args.runs
    
    for i in range(runs):
        all_results[i] = {}  # Run level
        # for ww in [1, 2, 3]:
        for ww in range(1, args.tree_width +1):
            all_results[i][ww] = {}  # Width level
            for d in [1, 2, 3, 4, 5]:
                print(f"Run {i}, Width {ww}, Depth {d}")
                args.tree_width = ww
                args.layers = d
                args.runs = i
                
                cos_sim, mse, l2_dist = main(args)
                
                # Calculate bottom 10% statistics
                cos_bottom_10_threshold = np.percentile(cos_sim.numpy(), 10)
                cos_bottom_10_values = cos_sim.numpy()[cos_sim.numpy() <= cos_bottom_10_threshold]
                mse_bottom_10_threshold = np.percentile(mse.numpy(), 10)
                mse_bottom_10_values = mse.numpy()[mse.numpy() <= mse_bottom_10_threshold]
                l2_dist_bottom_10_threshold = np.percentile(l2_dist.numpy(), 10)
                l2_dist_bottom_10_values = l2_dist.numpy()[l2_dist.numpy() <= l2_dist_bottom_10_threshold]
                # Store all metrics together for this configuration
                all_results[i][ww][d] = {
                    'cosine_bottom_10': cos_bottom_10_values,
                    'cosine_sim_mean': cos_sim.mean().item(),
                    'cosine_percentiles': np.percentile(cos_sim, [2.5, 97.5]),
                    'mse_mean': mse.mean().item(),
                    'mse_percentiles': np.percentile(mse, [2.5, 97.5]),
                    'mse_bottom_10': mse_bottom_10_values,
                    'l2_dist_mean': l2_dist.mean().item(),
                    'l2_dist_bottom_10': l2_dist_bottom_10_values,
                    'l2_dist_percentiles': np.percentile(l2_dist, [2.5, 97.5])
                }
                ch.save(cos_sim, f'cosim_d{d}_w{ww}_run{i}_{exp_id}.pt')
                ch.save(l2_dist, f'l2_dist_d{d}_w{ww}_run{i}_{exp_id}.pt')
                ch.save(mse, f'mse_d{d}_w{ww}_run{i}_{exp_id}.pt')

    # Save the entire structure
    ch.save(all_results, f'all_results_{exp_id}.pt')

    # Separate plot for each width
    # for width in range(1, 6):
    #     plt.figure(figsize=(10, 6))
    #     for depth in range(1, 6):
    #         lower, upper = all_results[width][depth]['cosine_similarity'].numpy()
    #         plt.plot([depth, depth], [lower, upper], marker='o')
    #     plt.xticks(range(1, 6), [f'{i} Layers' for i in range(1, 6)])
    #     plt.title(f'5th to 95th Percentile Range vs Depth (Width {width})')
    #     plt.xlabel('Depth (Layers)')
    #     plt.ylabel('Cosine Similarity')
    #     plt.grid(True)
    #     plt.show()
