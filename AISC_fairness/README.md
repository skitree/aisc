# AISC Fairness Experiments

There are two python scripts, base.py and waterbirds_data_structure.py; the first includes various fairness loss functions, while the second contains a class for data preprocessing, formatting, etc.

`fairness_exp_v0.ipynb` is an older notebook for experiments. Model checkpoints, val and training losses, learning rates, predictions, etc. can be found on mslurm.  

## Waterbirds class
### Functions
` __init__(self, metadata, images_dir, transform=None, split=None, split_ratios=(0.65, 0.15, 0.10, 0.1), pairing_distribution=None, seed=None, preload=False, device='cpu') >`
`split_ratios` sets the training, validation, test, and finetuning proportions. If None, we use the pre-set splits. Otherwise we re-shuffle and split (according to desired proportion and distributions)

`pairing_dist` specifies the distributions of pairings within the dataset; here, 40% of data should be water bird on water, land on land etc. 
if `pairing_distribution` is not specified in call, dataset splits are substantially skewed.
When specified, `pairing_distributions` will up or downsample based on desired distribution of class (bird), background subset pairings.

`__len__(self)` and `__getitem__(self, idx)` are straightforward.

`get_split(self, new_split, **kwargs)`Returns the subset of the dataset for the specified split (e.g., 0: train, 1: val, 2: test).

`check_class_background_pairings(self, split)` returns distributions of pairs (label, background) for specified split

`updating_target` and `shuffle_split` are helper functions for data processing.

## Base.py
This file contains the following fairness loss functions:
`equalized_loss`,
`parity_loss`,
`affirmative_action_modified_loss`,
`equalized_fpr_loss`,
`equalized_tpr_loss`, all of which have the following params `(predictions, labels, group_labels, device="cuda", alpha=0.1)`.

It also contains `validate_and_save_predictions(model, fairness, val_loader, criterion, file_info=None)`, which is used during the training loops. 

