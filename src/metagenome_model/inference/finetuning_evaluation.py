import os.path
import pickle
from tqdm import tqdm
import pandas as pd
import torch
from sklearn.metrics import auc, classification_report, matthews_corrcoef, precision_recall_curve, roc_auc_score

from src.metagenome_model.basic.kernel import Kernel
from src.metagenome_model.basic.metagenome_dataset import MetaGenomeSortSEQLengthForFinetuneDataset
from src.metagenome_model.models.finetune.finetuning_model import MetaOrionForPhenotype


class MetaOrionPhenotypeInfer(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = self.register_dataloader(custom_dataset=MetaGenomeSortSEQLengthForFinetuneDataset,
                                                   is_inference=True)
        self.model = MetaOrionForPhenotype(model_name_or_path=self.model_name_or_path, dropout_rate=self.dropout_rate,
                                           is_inference=True)
        self.figs_path = self.register_dir(os.path.join(self.output_home, 'figs'))
        self.probs_path = self.register_dir(os.path.join(self.output_home, 'probs'))
        self.PAN_LABELS = {
            'healthy': 0,
            'IBD': 1,
            'CRC': 2,
            'T2D': 3,
            'metabolic_syndrome': 4,
            'others': 5,
            'AS': 6,
            'OB': 7,
            'IBS': 8,
            'IGT': 9,
            'BL': 10,
            'ACVD': 11,
            'CKD': 12,
            'COVID-19': 13,
            'adenoma': 14,
            'melanoma': 15
        }

        if self.accelerator.is_main_process:
            self.accelerator.print(self.model)
            self.print_trainable_parameters()

    @torch.no_grad()
    def inference(self, **kwargs):
        self.model, self.dataloader = self.accelerator.prepare(self.model, self.dataloader)
        self.model.eval()
        batch_iters = tqdm(self.dataloader) if self.accelerator.is_main_process else self.dataloader
        total_loss = []
        all_pred = []
        all_label = []
        all_logit = []
        all_embed = []
        all_state_pred = []
        all_state_label = []
        all_state_logit = []

        all_filenames = []
        with torch.no_grad():
            for idx, sample in enumerate(batch_iters):
                output = self.model(
                    input_ids=sample['input_ids'],
                    attention_mask=sample['attention_mask'],
                    padding_mask=sample['padding_mask'],
                    abundance=sample['bin_abundance'],
                    age=sample['batch_age'],
                    gender=sample['batch_gender']
                )

                pan_label2id = {v: k for k, v in self.PAN_LABELS.items()}
                seq_len = sample['padding_mask'].sum(1)
                for sample_id in range(sample['input_ids'].shape[0]):
                    raw_label = sample['batch_label'].cpu()[sample_id].item()
                    label_name = pan_label2id[raw_label] if raw_label >= 0 else raw_label
                    taxa_emb = output.taxa_emb[sample_id][:seq_len[sample_id].item()].cpu().numpy()
                    taxa = sample['batch_seq'][sample_id].split(' ')
                    bin_abu = sample['bin_abundance'][sample_id][:seq_len[sample_id].item()].cpu().numpy()
                    os.makedirs(os.path.join(self.output_home, 'emb'), exist_ok=True)
                    with open(os.path.join(self.output_home, 'emb', sample['batch_filenames'][sample_id].split('.json')[0]+'.pkl'), 'wb') as fp:
                        pickle.dump({
                            'taxa': taxa,
                            'embedding': taxa_emb,
                            'binned_abundance': bin_abu,
                            'finetune_embedding': output.fusion_emb[sample_id].cpu().numpy(), # For visualization.
                            'label': label_name,
                        }, fp)

                pred = output.logits.softmax(dim=-1).argmax(dim=-1)
                state_pred = (output.state_logits.flatten().sigmoid() >= 0.5).long()

                all_pred.append(self.accelerator.gather_for_metrics(pred).detach().cpu())
                all_label.append(self.accelerator.gather_for_metrics(sample['batch_label']).detach().cpu())
                all_logit.append(self.accelerator.gather_for_metrics(output.logits).detach().cpu())

                all_state_label.append(self.accelerator.gather_for_metrics(sample['batch_state']).detach().cpu())
                all_state_pred.append(self.accelerator.gather_for_metrics(state_pred).detach().cpu())
                all_state_logit.append(
                    self.accelerator.gather_for_metrics(output.state_logits.flatten()).detach().cpu())

                all_embed.append(self.accelerator.gather_for_metrics(output.sample_emb).detach().cpu())
                all_filenames.append(self.accelerator.gather_for_metrics(sample['batch_filenames']))
        self.accelerator.end_training()

        if self.accelerator.is_main_process:
            preds = torch.cat(all_pred)
            labels = torch.cat(all_label)
            logits = torch.cat(all_logit)
            proba = logits.softmax(dim=-1).numpy()

            state_preds = torch.cat(all_state_pred)
            state_labels = torch.cat(all_state_label)
            state_logits = torch.cat(all_state_logit)
            state_proba = state_logits.sigmoid().numpy()

            name = [j.split('.json')[0] for i in all_filenames for j in i][:len(state_preds)]
            has_labels = bool((labels >= 0).all() and (state_labels >= 0).all())

            pan_prob_df = pd.DataFrame({
                'sample': name,
                'prob': state_proba,
                'pred': state_preds.numpy(),
            })
            multi_prob_df = pd.DataFrame(proba)
            multi_prob_df.columns = list(self.PAN_LABELS.keys())[1:]
            multi_prob_df.index = name

            if has_labels:
                pan_prob_df['label'] = state_labels.numpy()

            final_preds = [0 if state_preds[i] == 0 else preds[i].item() + 1 for i in range(len(state_preds))]
            multi_prob_df['pred'] = final_preds
            if has_labels:
                multi_prob_df['label'] = labels.numpy()

            pan_prob_df.to_csv(os.path.join(self.probs_path, 'pandisease.all.prob.csv'), index=False)
            multi_prob_df.to_csv(os.path.join(self.probs_path, 'multidisease.all.prob.csv'))

            if not has_labels:
                print('No labels found. Probability files were saved without metrics.')
                return None, None

            AUC = roc_auc_score(state_labels.numpy(), state_proba)
            precision, recall, thresholds = precision_recall_curve(state_labels.numpy(), state_proba)
            AUPR = auc(recall, precision)
            MCC = matthews_corrcoef(state_labels.numpy(), state_preds.numpy())
            report = classification_report(y_true=state_labels.numpy(), y_pred=state_preds.numpy(), output_dict=True)
            result = pd.DataFrame(report).transpose()
            result['auc'] = AUC
            result['aupr'] = AUPR
            result['MCC'] = MCC
            result.round(4).to_csv(os.path.join(self.output_home, 'pandisease.all.result.csv'))
            print('pandisease metrics\n', result)

            MCC = matthews_corrcoef(labels.numpy(), final_preds)
            report = classification_report(y_true=labels.numpy(), y_pred=final_preds, output_dict=True)
            multi_result = pd.DataFrame(report).transpose()
            multi_result['MCC'] = MCC
            multi_result.index = [{v: k for k, v in self.PAN_LABELS.items()}[int(i)] for i in multi_result.index[:-3]] + list(multi_result.index[-3:])
            multi_result.round(4).to_csv(os.path.join(self.output_home, 'multidisease.all.result.csv'))
            print('multidisease metrics\n', multi_result)

            return [result.loc['weighted avg', 'precision'], result.loc['weighted avg', 'recall'],
                    result.loc['weighted avg', 'f1-score'],
                    result.loc['accuracy', 'support'], result.loc['weighted avg', 'auc'],
                    result.loc['weighted avg', 'aupr'],
                    result.loc['weighted avg', 'MCC']], [multi_result.loc['weighted avg', 'precision'],
                                                         multi_result.loc['weighted avg', 'recall'],
                                                         multi_result.loc['weighted avg', 'f1-score'],
                                                         multi_result.loc['accuracy', 'support'],
                                                         multi_result.loc['weighted avg', 'MCC']]

        return None, None

    def train(self, **kwargs):
        pass
