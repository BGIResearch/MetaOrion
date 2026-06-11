# @Date    : 2026/6/10
# @Email   : zhangkexin2@genomics.cn

import os
import random
import warnings
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ==========================================
# 1. Global Configurations
# ==========================================
PAN_LABELS = {
    'healthy': 0, 'IBD': 1, 'CRC': 2, 'T2D': 3, 'metabolic_syndrome': 4,
    'others': 5, 'AS': 6, 'OB': 7, 'IBS': 8, 'IGT': 9, 'BL': 10,
    'ACVD': 11, 'CKD': 12, 'COVID-19': 13, 'adenoma': 14, 'melanoma': 15
}


def seed_torch(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ==========================================
# 2. Data Processing Functions
# ==========================================
def filter_profile(X):
    """Filter abundance profile, keeping only species-level (s__) features and removing all-zero columns."""
    index = [i for i, col in enumerate(X.columns) if col.split('|')[-1].split('__')[0] == 's']
    X = X.iloc[:, index]
    X.columns = [col.split('|')[-1] for col in X.columns]
    X = X.loc[:, (X != 0).any()]
    return X


def load_sample_list(file_path):
    """Read and format the split sample lists."""
    with open(file_path, 'r') as f:
        return sorted(set([os.path.splitext(os.path.basename(i.strip()))[0] for i in f.readlines()]))


def load_global_data(profile_path, label_path, task_type):
    """
    [Performance Optimization]
    Read the profile and label dictionary globally only once to share across all stages (train/val/test).
    """
    print(">>> Loading global profile and labels (this only happens once)...")

    # Load feature profile
    X_global = filter_profile(pd.read_csv(profile_path, sep='\t', index_col=0).transpose())

    # Load metadata/labels
    label_df = pd.read_csv(label_path, sep='\t', index_col=0)
    class_counts = label_df['disease_united'].value_counts()

    # Group rare diseases into 'others' based on a threshold
    disease_list = list(class_counts[class_counts >= 120].index)
    label_df.loc[~label_df['disease_united'].isin(disease_list), 'disease_united'] = 'others'

    # Generate global label dictionary
    label_dict = {}
    for i in label_df.index:
        dis = label_df.loc[i, 'disease_united']
        if task_type == 'pandisease':
            # Binary classification: healthy (0) vs disease (1)
            label_dict[str(i)] = 0 if dis == 'healthy' else 1
        else:
            # Multi-class classification
            label_dict[str(i)] = PAN_LABELS.get(dis, 5)

    return X_global, label_dict


def preprocess_indepdata_profile(test_path, profile_path, label_path, train_columns):
    """
    Process independent validation cohorts.
    Aligns features with the training set and fixes the task as binary classification.
    """
    test_profile = filter_profile(pd.read_csv(profile_path, sep='\t', index_col=0).transpose())

    # Pad the validation set according to the feature order of the training set (fill missing with 0)
    test_profile = test_profile.reindex(columns=train_columns, fill_value=0)

    test_paths = load_sample_list(test_path)
    label_df = pd.read_csv(label_path, sep='\t', index_col=0)

    # Independent cohorts are uniformly treated as binary classification (control vs disease)
    label_dict = {str(i): 0 if label_df.loc[i, 'study_condition'] == 'control' else 1 for i in label_df.index}

    return test_paths, test_profile, label_dict


# ==========================================
# 3. Dataset and Model
# ==========================================
class CustomDataset(Dataset):
    def __init__(self, data_paths, data, label_dict):
        self.data_paths = data_paths
        self.data = data
        self.label_dict = label_dict

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        name = self.data_paths[idx]
        return {
            'data': self.data.loc[name, :],
            'label': self.label_dict[name],
            'name': name
        }

    def collate_fn(self, batch):
        batch_data = [item['data'] for item in batch]
        batch_label = [item['label'] for item in batch]
        batch_name = [item['name'] for item in batch]
        return {
            'batch_data': torch.tensor(batch_data, dtype=torch.float32),
            'batch_label': torch.tensor(batch_label),
            'batch_name': batch_name
        }


class MLP(nn.Module):
    def __init__(self, num_classes, in_dim, hidden_dim=512):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 8),
            nn.Linear(hidden_dim // 8, hidden_dim // 8),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim // 8, num_classes)
        )
        self.classifier.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=1)

    def forward(self, x):
        return self.classifier(x)


# ==========================================
# 4. Core Evaluator Logic
# ==========================================
def calculate_loss_and_preds(logits, labels, task_type):
    """Unified loss calculation and prediction logic."""
    if task_type == 'pandisease':
        loss = F.binary_cross_entropy_with_logits(logits.flatten(), labels.float())
        probs = torch.sigmoid(logits.flatten())
        preds = (probs >= 0.5).long()
    else:
        loss = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
    return loss, probs, preds


def evaluate(model, dataloader, accelerator, task_type):
    """Universal evaluation function for Validation, Test, and Independent cohorts."""
    model.eval()
    all_loss, all_probs, all_preds, all_labels = [], [], [], []

    with torch.no_grad():
        for sample in dataloader:
            logits = model(sample['batch_data'])
            labels = sample['batch_label']

            loss, probs, preds = calculate_loss_and_preds(logits, labels, task_type)

            all_loss.append(accelerator.gather_for_metrics(loss).detach().cpu())
            all_probs.append(accelerator.gather_for_metrics(probs).detach().cpu())
            all_preds.append(accelerator.gather_for_metrics(preds).detach().cpu())
            all_labels.append(accelerator.gather_for_metrics(labels).detach().cpu())

    avg_loss = torch.mean(torch.hstack(all_loss)).item()
    labels_np = torch.hstack(all_labels).numpy()
    preds_np = torch.hstack(all_preds).numpy()

    result_dict = classification_report(y_true=labels_np, y_pred=preds_np, output_dict=True)
    metrics_df = pd.DataFrame(result_dict).transpose()

    # Calculate advanced metrics for binary classification
    if task_type == 'pandisease':
        probs_np = torch.hstack(all_probs).numpy()
        metrics_df['auc'] = roc_auc_score(labels_np, probs_np)
        precision, recall, _ = precision_recall_curve(labels_np, probs_np)
        metrics_df['aupr'] = auc(recall, precision)
        metrics_df['MCC'] = matthews_corrcoef(labels_np, preds_np)
    else:
        # For multi-class, probs is a 2D matrix; convert to list to prevent dimension errors when saving
        probs_np = torch.vstack(all_probs).numpy().tolist()

    return avg_loss, metrics_df, labels_np, preds_np, probs_np


# ==========================================
# 5. Independent Action Modules
# ==========================================
def train_model(train_path, val_path, X_global, label_dict, output_home, task_type, batch_size=48):
    """[Action 1] Exclusively handles training and saving the model."""
    print("\n>>> [Action] Training Model...")
    seed_torch(42)
    accelerator = Accelerator()

    train_paths = load_sample_list(train_path)
    val_paths = load_sample_list(val_path)
    random.shuffle(train_paths)
    random.shuffle(val_paths)

    train_loader = DataLoader(CustomDataset(train_paths, X_global, label_dict), batch_size=batch_size, shuffle=True,
                              collate_fn=CustomDataset(None, None, None).collate_fn)
    val_loader = DataLoader(CustomDataset(val_paths, X_global, label_dict), batch_size=batch_size,
                            collate_fn=CustomDataset(None, None, None).collate_fn)

    num_classes = 1 if task_type == 'pandisease' else len(PAN_LABELS)
    model = MLP(num_classes=num_classes, in_dim=X_global.shape[1], hidden_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    model, train_loader, val_loader, optimizer = accelerator.prepare(model, train_loader, val_loader, optimizer)

    min_loss = np.Inf
    patience, max_patience = 0, 20
    os.makedirs(output_home, exist_ok=True)
    model_save_path = os.path.join(output_home, 'model.bin')

    for eph in range(100):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f'Epoch: {eph + 1:03d}')

        for sample in batch_iterator:
            with accelerator.accumulate(model):
                logits = model(sample['batch_data'])
                loss, _, _ = calculate_loss_and_preds(logits, sample['batch_label'], task_type)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    batch_iterator.set_postfix({'Train_Loss': f'{loss.item():.4f}'})

        val_loss, metrics_df, _, _, _ = evaluate(model, val_loader, accelerator, task_type)

        if accelerator.is_main_process:
            if val_loss < min_loss:
                min_loss = val_loss
                print(
                    f"  -> Saved {eph + 1}-th epoch model. Val F1-Score: {metrics_df.loc['accuracy', 'f1-score']:.4f}")
                patience = 0
                accelerator.save(accelerator.unwrap_model(model).state_dict(), model_save_path)
            else:
                patience += 1

            if patience > max_patience:
                print("  -> Early stopping triggered.")
                break
    print(">>> Training Completed.")


def infer_test_set(test_path, X_global, label_dict, output_home, task_type, batch_size=48):
    """[Action 2] Exclusively handles inference on the equivalently distributed test set."""
    print("\n>>> [Action] Inference on Target Test Set...")
    seed_torch(42)
    accelerator = Accelerator()

    test_paths = load_sample_list(test_path)
    test_loader = DataLoader(CustomDataset(test_paths, X_global, label_dict), batch_size=batch_size,
                             collate_fn=CustomDataset(None, None, None).collate_fn)

    num_classes = 1 if task_type == 'pandisease' else len(PAN_LABELS)
    model = MLP(num_classes=num_classes, in_dim=X_global.shape[1], hidden_dim=512)

    model_save_path = os.path.join(output_home, 'model.bin')
    model.load_state_dict(torch.load(model_save_path))
    model, test_loader = accelerator.prepare(model, test_loader)

    _, test_metrics, test_labels, test_preds, test_probs = evaluate(model, test_loader, accelerator, task_type)

    if accelerator.is_main_process:
        print(test_metrics)
        os.makedirs(f"{output_home}/report", exist_ok=True)
        test_metrics.to_csv(f"{output_home}/report/result.csv")

        prob_df = pd.DataFrame({'sample': test_paths, 'prob': test_probs, 'pred': test_preds, 'label': test_labels})
        prob_df.to_csv(f"{output_home}/report/MLP_probs.csv", index=False)
        print(">>> Test Inference Completed and Saved.")


def infer_independent_cohort(test_path, profile_path, label_path, train_columns, model_path, save_dir, project_name):
    """[Action 3] Handles inference on independent validation cohorts."""
    print(f"\n>>> [Action] Inference on Independent Cohort: {project_name}...")
    seed_torch(42)
    accelerator = Accelerator()

    test_paths, data, label_dict = preprocess_indepdata_profile(test_path, profile_path, label_path, train_columns)

    test_loader = DataLoader(CustomDataset(test_paths, data, label_dict), batch_size=48,
                             collate_fn=CustomDataset(None, None, None).collate_fn)

    model = MLP(num_classes=1, in_dim=len(train_columns), hidden_dim=512)
    model.load_state_dict(torch.load(model_path))
    model, test_loader = accelerator.prepare(model, test_loader)

    _, metrics_df, labels, preds, probs = evaluate(model, test_loader, accelerator, task_type='pandisease')

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        print(metrics_df)

        metrics_df.to_csv(f"{save_dir}{project_name}_profile_mlp_result.csv")
        prob_df = pd.DataFrame({'sample': test_paths, 'prob': probs, 'pred': preds, 'label': labels})
        prob_df.to_csv(f"{save_dir}{project_name}_profile_mlp_probs.csv", index=False)


# ==========================================
# 6. Main Execution Block
# ==========================================
if __name__ == '__main__':
    # =======================================================
    # 🎯 Execution Switches (Modify to control execution flow)
    # =======================================================
    RUN_TRAINING = False  # Whether to execute model training
    RUN_TEST_INFERENCE = True  # Whether to execute test set inference
    RUN_INDEP_INFERENCE = False  # Whether to execute independent cohort inference

    TASKS = ['pandisease']  # Options: ['pandisease', 'multidisease']
    # =======================================================

    # Base global paths
    TRAIN_PROFILE_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v1121.train_test.profile'
    TRAIN_LABEL_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe'

    for task in TASKS:
        print(f"\n{'=' * 50}\nTarget Task: {task.upper()}\n{'=' * 50}")

        # 🚀 [Core Optimization] Load the massive table globally only once!
        X_global, label_dict_global = load_global_data(TRAIN_PROFILE_PATH, TRAIN_LABEL_PATH, task)

        for i in range(1, 6):
            split = f'split{i}'
            print(f"\n--- Processing {split} ---")

            data_dir = f'/bgi-seq-model-2/datasets/zhangkexin/meta_index/preprocess/metaphlan4/fine-tune/nov.specific.random.5/{split}/'
            output_home = f'/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/MLP-model/12.1/{task}/{split}/'

            train_path = data_dir + 'datapath.pandisease.train'
            val_path = data_dir + 'datapath.pandisease.val'
            test_path = data_dir + 'datapath.pandisease.test'

            # -----------------------------------
            # Action 1: Train model
            # -----------------------------------
            if RUN_TRAINING:
                train_model(train_path, val_path, X_global, label_dict_global, output_home, task, batch_size=48)

            # -----------------------------------
            # Action 2: Inference on test set
            # -----------------------------------
            if RUN_TEST_INFERENCE:
                infer_test_set(test_path, X_global, label_dict_global, output_home, task, batch_size=48)

            # -----------------------------------
            # Action 3: Inference on independent validation set (Binary classification only)
            # -----------------------------------
            # if RUN_INDEP_INFERENCE and task == 'pandisease':
            #     indepdata_base = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/'
            #     indep_save_dir = f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/MLP-model/indep.test/12.1/suppl/{split}/'
            #     model_bin_path = os.path.join(output_home, 'model.bin')
            #
            #     validation_cohorts = ['FanY_2023.anorexia', 'XuQ_2021.htn.t2d', 'iMSMS_2022.MS']
            #
            #     for disease_proj in validation_cohorts:
            #         test_idx_path = f'/bgi-seq-model-2/datasets/zhangkexin/meta_index/preprocess/metaphlan4/fine-tune/1.18.validation.supp/datapath.{disease_proj.split(".")[0]}.indeptest'
            #         prof_file = str(list(Path(indepdata_base + disease_proj).glob("*.profile"))[0])
            #         info_file = str(list(Path(indepdata_base + disease_proj).glob("*.metadata*"))[0])
            #
            #         infer_independent_cohort(
            #             test_idx_path, prof_file, info_file,
            #             train_columns=X_global.columns.tolist(),
            #             # Pass training set column names directly for alignment
            #             model_path=model_bin_path,
            #             save_dir=indep_save_dir,
            #             project_name=disease_proj
            #         )