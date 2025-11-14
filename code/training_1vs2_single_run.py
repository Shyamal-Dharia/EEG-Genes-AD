from functions import load_data, train, eval, get_valid_subjects
from models import conv_model
import torch
import torch.nn as nn
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                             recall_score, roc_auc_score, confusion_matrix)
import numpy as np
from torch.optim.lr_scheduler import StepLR

def seed_everything(seed=41):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === Configuration ===
data_directory = './HFD_PSD_stats_features_40sec'
valid_classes = {1, 2}
EPOCHS = 50
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
seq_len = 127            # fixed sequence length required for the flattening step
dropout = 0.05           # no dropout
num_classes = 2
batch_size = 45
seed_value = 2           # We only run once, so pick one seed

# === Set the seed and get valid subjects ===
seed_everything(seed_value)
valid_subjects = get_valid_subjects(data_directory, selected_classes=valid_classes)
print("Number of valid subjects:", len(valid_subjects))

# === Lists for storing combined predictions (for overall sample-level metrics) ===
run_all_sample_preds = []
run_all_sample_labels = []
run_all_sample_probs = []
run_all_sample_weights = []

# === Lists for subject-level majority-vote predictions ===
run_subject_level_preds = []
run_subject_level_actuals = []
run_subject_scores = []  # majority vote percentages per subject

# === Container for subject-level (per-subject sample) metrics ===
subject_metrics_list = []

# === Also keep aggregated data (gradients, features, etc.) if you need them ===
aggregated_subject_gradients = {}
aggregated_subject_cnn_features = {}
aggregated_subject_scores = {}

# =====================================================================
#                           MAIN LOOP
# =====================================================================
for subject in valid_subjects:
    print(f"\nProcessing subject: {subject}")
    try:
        train_loader, test_loader = load_data(
            target_participant=subject,
            directory=data_directory,
            selected_classes=valid_classes,
            batch_size=batch_size,
            shuffle=True
        )
    except ValueError as e:
        print(e)
        continue

    # Optional: Display one batch's shape for debugging.
    for batch in test_loader:
        data_tensor, label_tensor = batch
        print(f"  Test Loader batch shapes: {data_tensor.shape}, {label_tensor}")
        break

    # Determine input size (if needed).
    sample = next(iter(train_loader))[0]
    input_size = sample.view(sample.size(0), -1).shape[1]

    # === Initialize the model, optimizer, and loss function. ===
    model = conv_model(num_channels=seq_len, dropout=dropout, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    class_weights = torch.tensor([1.0, 1.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # === Train the model ===
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_acc = eval(model, test_loader, loss_fn, device)
        # (Optional: print progress)

    # === Evaluation Phase ===
    model.eval()
    subject_preds = []
    subject_probs = []
    subject_labels = []
    subject_weights = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs, wei = model(X)  # outputs: logits; wei: attention/feature maps
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            subject_preds.append(preds.cpu())
            subject_probs.append(probs[:, 1].cpu())  # positive class probabilities
            subject_labels.append(y.cpu())
            subject_weights.append(wei.cpu())

    subject_preds = torch.cat(subject_preds)
    subject_probs = torch.cat(subject_probs)
    subject_labels = torch.cat(subject_labels)

    try:
        subject_weights = torch.cat(subject_weights)
    except Exception as e:
        pass

    # === Save sample-level predictions from this subject (for overall sample-level metrics) ===
    run_all_sample_preds.extend(subject_preds.tolist())
    run_all_sample_labels.extend(subject_labels.tolist())
    run_all_sample_probs.extend(subject_probs.tolist())

    if isinstance(subject_weights, torch.Tensor):
        mean_weights = subject_weights.mean(dim=0).tolist()
    else:
        mean_weights = [w.mean().item() for w in subject_weights]
    run_all_sample_weights.extend(mean_weights)

    # === Subject-level majority vote ===
    majority_vote = torch.mode(subject_preds)
    final_pred = majority_vote.values.item()
    majority_count = (subject_preds == final_pred).sum().item()
    total_samples = len(subject_preds)
    majority_percentage = majority_count / total_samples

    # The actual class of this subject (assuming the test set truly has that one label):
    actual_class = test_loader.dataset[0][1].item()
    run_subject_level_actuals.append(actual_class)
    run_subject_level_preds.append(final_pred)
    run_subject_scores.append(majority_percentage)

    # === Compute sample-level metrics for this subject alone (optional) ===
    subj_accuracy  = accuracy_score(subject_labels, subject_preds)
    subj_f1        = f1_score(subject_labels, subject_preds, average='macro')
    subj_precision = precision_score(subject_labels, subject_preds)
    subj_recall    = recall_score(subject_labels, subject_preds)
    try:
        subj_roc_auc = roc_auc_score(subject_labels, subject_probs)
    except ValueError:
        subj_roc_auc = np.nan  # In case subject_labels are all same class

    subject_metrics_list.append({
        "subject_id": subject,
        "accuracy":   subj_accuracy,
        "f1":         subj_f1,
        "precision":  subj_precision,
        "recall":     subj_recall,
        "roc_auc":    subj_roc_auc
    })

    # === Compute Gradients and CNN Features for this subject (if desired) ===
    subject_gradients = []
    subject_cnn_features = []
    for X, y in test_loader:
        X = X.to(device)
        outputs, cnn_features = model(X)
        loss = outputs[:, actual_class].sum()  # focusing on 'actual_class'
        model.zero_grad()
        loss.backward()
        # We assume model.gradients is set up in your conv_model_1.
        grads_batch = model.gradients.clone().detach().cpu()
        cnn_features_batch = cnn_features.clone().detach().cpu()

        subject_gradients.append(grads_batch)
        subject_cnn_features.append(cnn_features_batch)

    subject_gradients = torch.cat(subject_gradients, dim=0)
    subject_cnn_features = torch.cat(subject_cnn_features, dim=0)
    print(f"Collected gradients for subject {subject}: shape {subject_gradients.shape}")

    # === Aggregate gradients, CNN features, majority-vote scores in your dicts ===
    if subject not in aggregated_subject_gradients:
        aggregated_subject_gradients[subject] = []
        aggregated_subject_cnn_features[subject] = []
        aggregated_subject_scores[subject] = []
    aggregated_subject_gradients[subject].append(subject_gradients)
    aggregated_subject_cnn_features[subject].append(subject_cnn_features)
    aggregated_subject_scores[subject].append(majority_percentage)


# =====================================================================
#                OVERALL SAMPLE-LEVEL METRICS (Single Run)
# =====================================================================
sample_accuracy  = accuracy_score(run_all_sample_labels, run_all_sample_preds)
sample_f1        = f1_score(run_all_sample_labels, run_all_sample_preds, average='macro')
sample_precision = precision_score(run_all_sample_labels, run_all_sample_preds)
sample_recall    = recall_score(run_all_sample_labels, run_all_sample_preds)
sample_roc_auc   = roc_auc_score(run_all_sample_labels, run_all_sample_probs)
sample_confusion = confusion_matrix(run_all_sample_labels, run_all_sample_preds)

print("\n=== Overall Sample-Level Metrics (Single Run) ===")
print(f"Accuracy:  {sample_accuracy*100:.2f}%")
print(f"F1 Score:  {sample_f1*100:.2f}%")
print(f"Precision: {sample_precision*100:.2f}%")
print(f"Recall:    {sample_recall*100:.2f}%")
print(f"ROC AUC:   {sample_roc_auc:.4f}")
print("Confusion Matrix:\n", sample_confusion)

# =====================================================================
#       SUBJECT-LEVEL (Majority Vote) METRICS (Single Run, Aggregated)
# =====================================================================
# The ROC AUC here is computed across all subjects using the majority-vote confidence scores:
subject_accuracy  = accuracy_score(run_subject_level_actuals, run_subject_level_preds)
subject_f1        = f1_score(run_subject_level_actuals, run_subject_level_preds, average='macro')
subject_precision = precision_score(run_subject_level_actuals, run_subject_level_preds)
subject_recall    = recall_score(run_subject_level_actuals, run_subject_level_preds)
try:
    subject_roc_auc = roc_auc_score(run_subject_level_actuals, run_subject_scores)
except ValueError:
    subject_roc_auc = np.nan
subject_confusion = confusion_matrix(run_subject_level_actuals, run_subject_level_preds)

print("\n=== Subject-Level (Majority Vote) Metrics (Single Run) ===")
print(f"Accuracy:  {subject_accuracy*100:.2f}%")
print(f"F1 Score:  {subject_f1*100:.2f}%")
print(f"Precision: {subject_precision*100:.2f}%")
print(f"Recall:    {subject_recall*100:.2f}%")
print(f"ROC AUC:   {subject_roc_auc:.4f}  (Aggregated across subjects)")
print("Confusion Matrix:\n", subject_confusion)

# =====================================================================
#  MEAN ± STD OF PER-SUBJECT SAMPLE-LEVEL METRICS (Optional Extra)
# =====================================================================
per_subj_acc  = [m["accuracy"]  for m in subject_metrics_list]
per_subj_f1   = [m["f1"]        for m in subject_metrics_list]
per_subj_prec = [m["precision"] for m in subject_metrics_list]
per_subj_rec  = [m["recall"]    for m in subject_metrics_list]
per_subj_roc  = [m["roc_auc"]   for m in subject_metrics_list if not np.isnan(m["roc_auc"])]

mean_acc,  std_acc  = np.mean(per_subj_acc),  np.std(per_subj_acc)
mean_f1,   std_f1   = np.mean(per_subj_f1),   np.std(per_subj_f1)
mean_prec, std_prec = np.mean(per_subj_prec), np.std(per_subj_prec)
mean_rec,  std_rec  = np.mean(per_subj_rec),  np.std(per_subj_rec)

print("\n=== PER-SUBJECT Sample-Level Metrics: Mean ± Std (Single Run) ===")
print(f"Accuracy:  {mean_acc:.3f} ± {std_acc:.3f}")
print(f"F1 Score:  {mean_f1:.3f} ± {std_f1:.3f}")
print(f"Precision: {mean_prec:.3f} ± {std_prec:.3f}")
print(f"Recall:    {mean_rec:.3f} ± {std_rec:.3f}")

# =====================================================================
#     SAVE GRADIENTS, FEATURES, AND SAMPLE-LEVEL PREDICTIONS (if desired)
# =====================================================================
avg_subject_gradients   = {}
avg_subject_cnn_features = {}
avg_subject_scores       = {}

for subject in aggregated_subject_gradients:
    grads_list = aggregated_subject_gradients[subject]  # list with a single item per subject
    feats_list = aggregated_subject_cnn_features[subject]
    score_list = aggregated_subject_scores[subject]

    final_grads = grads_list[0]
    final_feats = feats_list[0]
    final_score = score_list[0]

    avg_subject_gradients[subject] = final_grads.numpy()
    avg_subject_cnn_features[subject] = final_feats.numpy()
    avg_subject_scores[subject] = final_score

np.savez(
    'single_run_gradients_and_scores_1vs2.npz',
    avg_subject_gradients     = avg_subject_gradients,
    avg_subject_cnn_features  = avg_subject_cnn_features,
    avg_subject_scores        = avg_subject_scores,
    sample_accuracy           = sample_accuracy,
    sample_f1                 = sample_f1,
    sample_precision          = sample_precision,
    sample_recall             = sample_recall,
    sample_roc_auc            = sample_roc_auc,
    sample_confusion          = sample_confusion,
    subject_accuracy          = subject_accuracy,
    subject_f1                = subject_f1,
    subject_precision         = subject_precision,
    subject_recall            = subject_recall,
    subject_roc_auc           = subject_roc_auc,
    subject_confusion         = subject_confusion,
    per_subject_metrics       = subject_metrics_list,
    sample_preds              = run_all_sample_preds,
    sample_labels             = run_all_sample_labels,
    sample_probs              = run_all_sample_probs
)

print("\nAll done! Saved results to 'single_run_gradients_and_scores_1vs2.npz'.")
