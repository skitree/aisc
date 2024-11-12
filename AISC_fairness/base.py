"""
Re-used functions for AISC fairness experiments. Includes fairness functions for demographic parity
and equalized odds. 
"""

def validate_and_save_predictions(model, fairness, val_loader, criterion, file_info=None):
    """
    Validate the model and save predictions to a file, accounting for fairness loss.
    
    Args:
        model: The trained model.
        val_loader: DataLoader for validation data.
        criterion: The primary classification loss (e.g., CrossEntropyLoss).
        fairness: The type of fairness constraint (e.g., "parity", "equalized").
    
    Returns:
        The average validation loss, the predictions, the true labels, and the fairness loss.
    """
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    running_fairness_loss = 0.0  # Track fairness loss
    all_preds = []
    all_labels = []
    
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
    
    Args:
        predictions (torch.Tensor): Model predictions (probabilities or logits).
        labels (torch.Tensor): True labels (binary classification: 0 or 1).
        group_labels (torch.Tensor): Sensitive attribute labels (e.g., 0 for land, 1 for water).
        alpha (float): Weight of the fairness penalty in the loss.
    
    Returns:
        torch.Tensor: The Equalized Odds loss.
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


"""
Demographic parity (group_labels is background / place; either water or land)
"""
def parity_loss(predictions, labels, group_labels, alpha=0.1):
    # Group labels can be the background type: 0 for land, 1 for water
    group_0_mask = (group_labels == 0)
    group_1_mask = (group_labels == 1)
    
    avg_group_0 = torch.mean(predictions[group_0_mask])
    avg_group_1 = torch.mean(predictions[group_1_mask])

    # Minimize the difference between the average predictions for each group
    parity_loss = torch.abs(avg_group_0 - avg_group_1)
    
    return alpha * parity_loss