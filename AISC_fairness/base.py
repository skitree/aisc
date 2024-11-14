"""
Re-used functions for AISC fairness experiments. Includes fairness functions. Fairness function params are 
(predictions, labels, group_labels, device="cuda", alpha=0.1)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import re
import os
import shutil
import numpy as np
from tqdm import tqdm

def validate_and_save_predictions(model, fairness, val_loader, criterion, file_info=None):
    """
    Validate the model and save predictions to a file, accounting for fairness loss.

    """
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    running_fairness_loss = 0.0  # Track fairness loss
    all_preds = []
    all_labels = []
    
    device = "cuda"
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for images, labels, species, place in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Compute primary validation loss
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            # Compute fairness loss if a fairness constraint is provided
            if fairness == "parity":
                group_labels = place  # Example of using place as the sensitive attribute
                fair_loss = parity_loss(outputs, labels, group_labels)
                running_fairness_loss += fair_loss.item()
            elif fairness == "equalized":
                group_labels = place
                fair_loss = equalized_loss(outputs, labels, group_labels)
                running_fairness_loss += fair_loss.item()
            
            # Collect predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute average losses
    avg_val_loss = running_val_loss / len(val_loader)
    avg_fairness_loss = running_fairness_loss / len(val_loader) if fairness else 0.0
    total_val_loss = avg_val_loss + avg_fairness_loss  # Combine primary and fairness losses

    # Save predictions and true labels to a file
    np.save(f'{file_info}_predictions.npy', np.array(all_preds))  # Save predictions
    np.save(f'{file_info}_true_labels.npy', np.array(all_labels))  # Save true labels
    
    return total_val_loss, avg_val_loss, avg_fairness_loss, all_preds, all_labels



def equalized_loss(predictions, labels, group_labels, device="cuda", alpha=0.1):
    """
    Computes the Equalized Odds loss to ensure that the predictions
    are conditionally independent of group_labels given the true labels.
    
    """
    
    # Move tensors to the correct device
    predictions = predictions.to(device)
    labels = labels.to(device)
    group_labels = group_labels.to(device)
    
    # Separate groups
    group_0_mask = (group_labels == 0)
    group_1_mask = (group_labels == 1)
    
    # Separate true positives (labels == 1) and false positives (labels == 0)
    tpr_0_mask = group_0_mask & (labels == 1)  # True positives in group 0
    tpr_1_mask = group_1_mask & (labels == 1)  # True positives in group 1
    
    fpr_0_mask = group_0_mask & (labels == 0)  # False positives in group 0
    fpr_1_mask = group_1_mask & (labels == 0)  # False positives in group 1
    
    # Compute the average predictions (mean of the probabilities) for each group for TPR and FPR
    tpr_0 = torch.mean(predictions[tpr_0_mask]) if tpr_0_mask.sum() > 0 else torch.tensor(0.0, device=device)
    tpr_1 = torch.mean(predictions[tpr_1_mask]) if tpr_1_mask.sum() > 0 else torch.tensor(0.0, device=device)
    
    fpr_0 = torch.mean(predictions[fpr_0_mask]) if fpr_0_mask.sum() > 0 else torch.tensor(0.0, device=device)
    fpr_1 = torch.mean(predictions[fpr_1_mask]) if fpr_1_mask.sum() > 0 else torch.tensor(0.0, device=device)

    # Minimize the difference in TPR and FPR across groups
    tpr_diff = torch.abs(tpr_0 - tpr_1)
    fpr_diff = torch.abs(fpr_0 - fpr_1)
    
    # Total Equalized Odds loss: difference in TPR and FPR between groups
    equalized_odds_loss = tpr_diff + fpr_diff
    
    return equalized_odds_loss



def parity_loss(predictions, labels, group_labels, alpha=0.1):
    """
    Demographic parity (group_labels is background / place; either water or land)
    """
    # Group labels can be the background type: 0 for land, 1 for water
    group_0_mask = (group_labels == 0)
    group_1_mask = (group_labels == 1)
    
    avg_group_0 = torch.mean(predictions[group_0_mask])
    avg_group_1 = torch.mean(predictions[group_1_mask])

    # Minimize the difference between the average predictions for each group
    parity_loss = torch.abs(avg_group_0 - avg_group_1)
    
    return alpha * parity_loss



def affirmative_action_modified_loss(predictions, labels, group_labels, device="cuda", alpha=0.1):
    """
    Computes a custom Affirmative Action loss to encourage TPR of at least 0.95 for group A=1 
    and at least 0.9 for group A=0.

    """
    
    # Move tensors to the correct device
    predictions = predictions.to(device)
    labels = labels.to(device)
    group_labels = group_labels.to(device)
    
    # Separate groups
    group_0_mask = (group_labels == 0)
    group_1_mask = (group_labels == 1)
    
    # Calculate TPR for each group
    # True positives for group 0 and group 1 (where labels == 1)
    tpr_0_mask = group_0_mask & (labels == 1)
    tpr_1_mask = group_1_mask & (labels == 1)
    
    # Total actual positives in each group
    actual_positives_0 = tpr_0_mask.sum().float()
    actual_positives_1 = tpr_1_mask.sum().float()
    
    # Calculate TPR for each group
    tpr_0 = predictions[tpr_0_mask].mean() if actual_positives_0 > 0 else torch.tensor(0.0, device=device)
    tpr_1 = predictions[tpr_1_mask].mean() if actual_positives_1 > 0 else torch.tensor(0.0, device=device)
    
    # Calculate the penalty for each group based on the target thresholds
    penalty_0 = torch.pow(torch.minimum(torch.tensor(0.0, device=device), 0.9 - tpr_0), 2)
    penalty_1 = torch.pow(torch.minimum(torch.tensor(0.0, device=device), 0.95 - tpr_1), 2)
    
    # Total Affirmative Action loss
    affirmative_action_loss = alpha * (penalty_1 - penalty_0)
    
    return affirmative_action_loss


def equalized_tpr_loss(predictions, labels, group_labels, device="cuda", alpha=0.1):
    """
    Computes the Equalized TPR loss to minimize the squared difference in TPR between
    two groups specified by group_labels.
    """
    
    # Move tensors to the correct device
    predictions = predictions.to(device)
    labels = labels.to(device)
    group_labels = group_labels.to(device)
    
    # Separate groups
    group_0_mask = (group_labels == 0)
    group_1_mask = (group_labels == 1)
    
    # Calculate True Positive Rate (TPR) for each group
    # True positives for group 0 and group 1 (where labels == 1)
    tpr_0_mask = group_0_mask & (labels == 1)
    tpr_1_mask = group_1_mask & (labels == 1)
    
    # Total actual positives in each group
    actual_positives_0 = group_0_mask & (labels == 1)
    actual_positives_1 = group_1_mask & (labels == 1)
    
    # Avoid division by zero by setting TPR to zero if no positives are present in a group
    tpr_0 = predictions[tpr_0_mask].sum() / actual_positives_0.sum().float() if actual_positives_0.sum() > 0 else torch.tensor(0.0, device=device)
    tpr_1 = predictions[tpr_1_mask].sum() / actual_positives_1.sum().float() if actual_positives_1.sum() > 0 else torch.tensor(0.0, device=device)
    
    # Calculate the squared difference in TPRs
    tpr_diff_squared = torch.pow(tpr_1 - tpr_0, 2)
    
    # Total Equalized TPR loss
    equalized_tpr_loss = alpha * tpr_diff_squared
    
    return equalized_tpr_loss

def equalized_fpr_loss(predictions, labels, group_labels, alpha=0.1, device="cuda"):
    """
    Computes the Equalized FPR loss to minimize the squared difference in FPR between
    two groups specified by group_labels.
    """
    
    # Move tensors to the correct device
    predictions = predictions.to(device)
    labels = labels.to(device)
    group_labels = group_labels.to(device)
    
    # Separate groups
    group_0_mask = (group_labels == 0)
    group_1_mask = (group_labels == 1)
    
    # Calculate False Positive Rate (FPR) for each group
    # False positives for group 0 and group 1 (where labels == 0)
    fpr_0_mask = group_0_mask & (labels == 0)
    fpr_1_mask = group_1_mask & (labels == 0)
    
    # Total actual negatives in each group
    actual_negatives_0 = group_0_mask & (labels == 0)
    actual_negatives_1 = group_1_mask & (labels == 0)
    
    # Avoid division by zero by setting FPR to zero if no negatives are present in a group
    fpr_0 = predictions[fpr_0_mask].mean() if actual_negatives_0.sum() > 0 else torch.tensor(0.0, device=device)
    fpr_1 = predictions[fpr_1_mask].mean() if actual_negatives_1.sum() > 0 else torch.tensor(0.0, device=device)
    
    # Calculate the squared difference in FPRs
    fpr_diff_squared = torch.pow(fpr_1 - fpr_0, 2)
    
    # Total Equalized FPR loss
    equalized_fpr_loss = alpha * fpr_diff_squared
    
    return equalized_fpr_loss

