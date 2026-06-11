import os
import json
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, classification_report, roc_auc_score, precision_recall_curve,
    auc, confusion_matrix, matthews_corrcoef
)


def plot_metrics(preds, labels, model_name, save_dir):
    """
    Plot and save the binary confusion matrix.
    """
    sns.set()
    f, ax = plt.subplots(figsize=(4, 4))
    plt.rc('font', size=16)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    print(f"Confusion Matrix for {model_name}:\n", cm)

    sns.heatmap(cm, annot=True, ax=ax, fmt='.20g', cmap='Blues', cbar=False)

    ax.set_xlabel('Predicted Label', fontsize=18)
    ax.set_ylabel('True Label', fontsize=18)

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Control', 'Case'], fontsize=16)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Control', 'Case'], fontsize=16)

    ax.tick_params(axis='both', which='both', direction='in', length=5, width=1.5)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name}.png"), dpi=800, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f"{model_name}.pdf"), bbox_inches='tight', format='pdf')
    plt.show()


def plot_multi_metrics(preds, labels, model_name, save_dir, classes=None, class_names=None):
    """
    Plot and save the multi-class confusion matrix, normalized by row but annotated with raw counts.
    """
    sns.set()
    f, ax = plt.subplots(figsize=(6, 6))
    plt.rc('font', size=16)

    if classes is None:
        classes = sorted(np.unique(np.concatenate([labels, preds])))
    if class_names is None:
        class_names = [str(c) for c in classes]
    elif len(class_names) != len(classes):
        raise ValueError("The length of class_names must match the number of classes.")

    # Calculate raw confusion matrix
    cm = confusion_matrix(labels, preds, labels=classes)
    print(f"Raw Confusion Matrix for {model_name}:\n", cm)

    # Normalize by row (calculate proportions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Draw heatmap using normalized values for color, but raw integers for text
    sns.heatmap(
        cm_normalized,
        annot=cm,
        fmt='d',
        ax=ax,
        cmap='Blues',
        cbar=False,
        annot_kws={"fontsize": 12}
    )

    ax.set_xlabel('Predicted Label', fontsize=18)
    ax.set_ylabel('True Label', fontsize=18)

    n_classes = len(classes)
    xticks_pos = np.arange(n_classes) + 0.5
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(class_names, fontsize=16, rotation=45, ha='right')
    ax.set_yticks(xticks_pos)
    ax.set_yticklabels(class_names, fontsize=16, rotation=0)
    ax.tick_params(axis='both', which='both', direction='in', length=5, width=1.5)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name}.png"), dpi=800, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f"{model_name}.pdf"), bbox_inches='tight', format='pdf')
    plt.show()


def preprocess_data_profile(datapath_dir, disease, profile_path, metadata_path, categories_mapping=None,
                            PAN_LABELS=None):
    """
    Load data splits, fetch phenotype labels, and construct feature matrices.
    Optimized for vectorized Pandas extraction.
    """

    def load_split_ids(split_type):
        path = os.path.join(datapath_dir, f'datapath.{disease}.{split_type}')
        with open(path, 'r') as f:
            return list(set([os.path.splitext(os.path.basename(i.strip()))[0] for i in f.readlines()]))

    train = load_split_ids('train')
    val = load_split_ids('val')
    test = load_split_ids('test')

    # Load Metadata
    label_df = pd.read_csv(metadata_path, sep='\t', index_col=0)

    # Assign labels based on classification type
    if categories_mapping is None:
        # Binary Classification
        label_dict = {i: 0 if label_df.loc[i, 'disease_united'] == 'healthy' else 1 for i in train + test + val if
                      i in label_df.index}
    else:
        # Multi-class Classification
        class_counts = label_df['disease_united'].value_counts()
        disease_list = list(class_counts[class_counts >= 120].index)
        label_df.loc[~label_df['disease_united'].isin(disease_list), 'disease_united'] = 'others'
        label_dict = {i: PAN_LABELS[label_df.loc[i, 'disease_united']] for i in train + test + val if
                      i in label_df.index}

    # Load Abundance Profile
    X = pd.read_csv(profile_path, sep='\t', index_col=0).transpose()

    # Filter for species-level taxa (s__)
    species_indices = [i for i in range(len(X.columns)) if X.columns[i].split('|')[-1].startswith('s__')]
    X = X.iloc[:, species_indices]
    feature_names = [x.split('|')[-1] for x in X.columns.tolist()]

    # Secure intersection (prevents KeyError if a sample is in the split list but missing from the dataframe)
    valid_train = [x for x in train if x in X.index and x in label_dict]
    valid_val = [x for x in val if x in X.index and x in label_dict]
    valid_test = [x for x in test if x in X.index and x in label_dict]

    # Vectorized data extraction (Massive Speedup)
    X_train, y_train = X.loc[valid_train].values, np.array([label_dict[x] for x in valid_train])
    X_val, y_val = X.loc[valid_val].values, np.array([label_dict[x] for x in valid_val])
    X_test, y_test = X.loc[valid_test].values, np.array([label_dict[x] for x in valid_test])

    print('Train Distribution:', Counter(y_train))
    print('Val Distribution:', Counter(y_val))
    print('Test Distribution:', Counter(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, valid_train, valid_val, valid_test


def train_ML_model(X_train, y_train, save_dir, disease, n_jobs=32):
    """
    Train Random Forest, Logistic Regression, and XGBoost models.
    """
    model_save_dir = os.path.join(save_dir, disease)
    os.makedirs(model_save_dir, exist_ok=True)

    print(">>> Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    with open(os.path.join(model_save_dir, 'RF.model.pkl'), 'wb') as f:
        pickle.dump(clf, f)

    print(">>> Training XGBoost...")
    xgb_model = xgb.XGBClassifier(booster='gbtree', n_estimators=500, seed=42, n_jobs=n_jobs)
    xgb_model.fit(X_train, y_train)
    with open(os.path.join(model_save_dir, 'XGBoost.model.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)

    print(">>> Training Logistic Regression...")
    log_reg = LogisticRegression(random_state=42, max_iter=1000, n_jobs=n_jobs)
    log_reg.fit(X_train, y_train)
    with open(os.path.join(model_save_dir, 'LR.model.pkl'), 'wb') as f:
        pickle.dump(log_reg, f)

    print('>>> Model training completed.')
    return clf, xgb_model, log_reg


def load_ML_model(save_dir, disease):
    """
    Load pre-trained ML models.
    """
    model_dir = os.path.join(save_dir, disease)
    with open(os.path.join(model_dir, 'RF.model.pkl'), 'rb') as f:
        clf = pickle.load(f)
    with open(os.path.join(model_dir, 'LR.model.pkl'), 'rb') as f:
        log_reg = pickle.load(f)
    with open(os.path.join(model_dir, 'XGBoost.model.pkl'), 'rb') as f:
        xgb_model = pickle.load(f)
    return clf, xgb_model, log_reg


def ML_model_predict(model, model_name, disease, X_test, y_test, test_sample, output_dir, split_name):
    """
    Evaluate binary classification models and save metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    AUC = roc_auc_score(y_test, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    AUPR = auc(recall, precision)
    MCC = matthews_corrcoef(y_test, y_pred)

    report = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
    result = pd.DataFrame(report).transpose()
    result['AUC'] = AUC
    result['AUPR'] = AUPR
    result['MCC'] = MCC

    save_dir = os.path.join(output_dir, disease, split_name)
    os.makedirs(save_dir, exist_ok=True)

    result.to_csv(os.path.join(save_dir, f'{disease}_profile_{model_name}_result.csv'))

    # Save Confusion Matrix Plot
    plot_metrics(y_pred, y_test, f'{disease}_{model_name}_confusion_matrix', save_dir)

    # Save Probabilities
    prob = pd.DataFrame({'sample': test_sample, 'prob': y_prob, 'pred': y_pred, 'label': y_test})
    prob.to_csv(os.path.join(save_dir, f'{disease}_profile_{model_name}_probs.csv'), index=False)

    print(f'>>> Evaluated {model_name} on {split_name} set.')


def ML_model_multi_predict(model, model_name, disease, X_test, y_test, test_sample, output_dir, split_name, PAN_LABELS):
    """
    Evaluate multi-class classification models and save metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    MCC = matthews_corrcoef(y_test, y_pred)

    report = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
    result = pd.DataFrame(report).transpose()
    result['MCC'] = MCC

    save_dir = os.path.join(output_dir, disease, split_name)
    os.makedirs(save_dir, exist_ok=True)

    result.to_csv(os.path.join(save_dir, f'{disease}_profile_{model_name}_result.csv'))

    plot_multi_metrics(
        y_pred, y_test, f'{disease}_multiclass_{model_name}_confusion_matrix', save_dir,
        classes=list(PAN_LABELS.values()), class_names=list(PAN_LABELS.keys())
    )

    prob = pd.DataFrame(y_prob, columns=list(PAN_LABELS.keys()), index=test_sample)
    prob['pred'] = y_pred
    prob['label'] = y_test
    prob.to_csv(os.path.join(save_dir, f'{disease}_profile_{model_name}_probs.csv'))

    print(f'>>> Evaluated {model_name} (Multi-class) on {split_name} set.')


if __name__ == '__main__':
    # ==========================================
    # Global Paths and Configuration
    # ==========================================
    PROFILE_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v1121.train_test.profile'
    METADATA_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe'

    DATASET_DIR = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/preprocess/metaphlan4/fine-tune/nov.specific.random.5/'
    SAVE_PATH = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/ML-model/nov.specific.random.5/'

    DISEASE = 'pandisease'
    PAN_LABELS_SPECIFIC = {
        'healthy': 0, 'IBD': 1, 'CRC': 2, 'T2D': 3, 'metabolic_syndrome': 4,
        'others': 5, 'AS': 6, 'OB': 7, 'IBS': 8, 'IGT': 9, 'BL': 10,
        'ACVD': 11, 'CKD': 12, 'COVID-19': 13, 'adenoma': 14, 'melanoma': 15
    }

    # ==========================================
    # Main Execution Loop
    # ==========================================
    for split_folder in os.listdir(DATASET_DIR):
        if 'split' not in split_folder:
            continue

        print(f"\n==========================================")
        print(f"Processing {split_folder}...")
        print(f"==========================================")

        datapath_dir = os.path.join(DATASET_DIR, split_folder, '')
        save_dir = os.path.join(SAVE_PATH, split_folder, '')

        # 1. Preprocess PanDisease Data
        X_train, X_val, X_test, y_train, y_val, y_test, _, train_ids, val_ids, test_ids = preprocess_data_profile(
            datapath_dir=datapath_dir,
            disease=DISEASE,
            profile_path=PROFILE_PATH,
            metadata_path=METADATA_PATH
        )

        # 2. Train/Load PanDisease Models
        # clf, xgb_model, log_reg = train_ML_model(X_train, y_train, save_dir, DISEASE)
        clf, xgb_model, log_reg = load_ML_model(save_dir, DISEASE)

        # 3. Evaluate PanDisease Models on Test Set
        ML_model_predict(clf, 'rf', DISEASE, X_test, y_test, test_ids, save_dir, 'test')
        ML_model_predict(xgb_model, 'xgb', DISEASE, X_test, y_test, test_ids, save_dir, 'test')
        ML_model_predict(log_reg, 'lr', DISEASE, X_test, y_test, test_ids, save_dir, 'test')

        # 4. Evaluate PanDisease Models on Val Set
        ML_model_predict(clf, 'rf', DISEASE, X_val, y_val, val_ids, save_dir, 'val')
        ML_model_predict(xgb_model, 'xgb', DISEASE, X_val, y_val, val_ids, save_dir, 'val')
        ML_model_predict(log_reg, 'lr', DISEASE, X_val, y_val, val_ids, save_dir, 'val')

        # 5. Preprocess Multi-Disease Data
        X_train_mc, X_val_mc, X_test_mc, y_train_mc, y_val_mc, y_test_mc, _, train_ids_mc, val_ids_mc, test_ids_mc = preprocess_data_profile(
            datapath_dir=datapath_dir,
            disease=DISEASE,
            profile_path=PROFILE_PATH,
            metadata_path=METADATA_PATH,
            categories_mapping=PAN_LABELS_SPECIFIC,
            PAN_LABELS=PAN_LABELS_SPECIFIC
        )

        # 6. Train/Load Multi-Disease Models
        # clf_mc, xgb_model_mc, log_reg_mc = train_ML_model(
        #     X_train_mc, y_train_mc, save_dir, 'multidisease'
        # )
        clf_mc, xgb_model_mc, log_reg_mc = load_ML_model(save_dir, DISEASE)

        # 7. Evaluate Multi-Disease Models on Test Set
        ML_model_multi_predict(clf_mc, 'rf', 'multidisease', X_test_mc, y_test_mc, test_ids_mc, save_dir, 'test',
                               PAN_LABELS_SPECIFIC)
        ML_model_multi_predict(xgb_model_mc, 'xgb', 'multidisease', X_test_mc, y_test_mc, test_ids_mc, save_dir, 'test',
                               PAN_LABELS_SPECIFIC)
        ML_model_multi_predict(log_reg_mc, 'lr', 'multidisease', X_test_mc, y_test_mc, test_ids_mc, save_dir, 'test',
                               PAN_LABELS_SPECIFIC)

        # 8. Evaluate Multi-Disease Models on Test Set
        ML_model_multi_predict(clf_mc, 'rf', 'multidisease', X_val_mc, y_val_mc, val_ids_mc, save_dir, 'val',
                               PAN_LABELS_SPECIFIC)
        ML_model_multi_predict(xgb_model_mc, 'xgb', 'multidisease', X_val_mc, y_val_mc, val_ids_mc, save_dir, 'val',
                               PAN_LABELS_SPECIFIC)
        ML_model_multi_predict(log_reg_mc, 'lr', 'multidisease', X_val_mc, y_val_mc, val_ids_mc, save_dir, 'val',
                               PAN_LABELS_SPECIFIC)

        print(f">>> Finished {split_folder}.")
