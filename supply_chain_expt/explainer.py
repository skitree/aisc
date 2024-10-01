import torch as ch
import numpy as np
from lime import lime_tabular

# Function that computes LIME explanations
def explain(model_fn, X, n_samples=500):
    # Create an explainer
    explainer = lime_tabular.LimeTabularExplainer(X, 
                                                  feature_names=["feature_{}".format(i) \
                                                                 for i in range(X.shape[1])],
                                                                 discretize_continuous=False,
                                                                 mode="regression")
        
    # Create an explainer
    explanations = [] 
    for i in range(X.shape[0]):
        exp = explainer.explain_instance(X[i], model_fn, num_features=X.shape[1], num_samples=n_samples)
        lime_exp = list(dict(sorted(dict(exp.as_list()).items())).values())
        explanations.append(lime_exp)
    
    return ch.tensor(explanations)

# Function that checks recorse distance 
def check_recourse_distance(explanation, model, X):
    # set targets that are 10% higher than the current prediction
    y = model(X)
    target = 0 # y + 0.1 * np.abs(y)
    # See how far in the explanation direction in X space we need to go to reach the target
    explanation = explanation / explanation.norm(dim=1).unsqueeze(-1)
    orig_X = X.clone()
    for _ in range(10_000):
        y = model(X)
        if (y > target).all():
            break
        # Print fraction of X's that are over the target
        # print((y > target).mean())
        X += (0.01 * explanation) * (y < target)
    
    # Return distances
    return np.linalg.norm(X - orig_X, axis=1)
