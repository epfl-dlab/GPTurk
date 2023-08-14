import pandas as pd
import torch
from dataloaders.dataloader_gpturk import GPTurkDataset

from config import SENTIMENT_DATA_DIR, MODELS_DIR, UBI_TOPIC_DATA_DIR, GPTURK_DATA_DIR, UBI_STANCE_DATA_DIR
from finetuning.custom_callbacks import CustomWandbCallback

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    hamming_loss,
    confusion_matrix
)

import numpy as np
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
 
def wandb_hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "objective", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
            "per_device_train_batch_size": {"values": [16, 32, 64, 128]},
        },
    }   

def compute_metrics_multi_label(eval_pred):
    logits, labels = eval_pred
    predictions = np.zeros(logits.shape)
    predictions[np.where(logits >= 0.2)] = 1    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    h_loss = hamming_loss(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'hamming_loss': h_loss      # https://stats.stackexchange.com/questions/233275/multilabel-classification-metrics-on-scikit
    }

@hydra.main(version_base=None, config_path="conf", config_name="finetuning_conf.yaml")
def main(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt)
    if cfg.dataset == "gpturk_transductive":
        dataset = GPTurkDataset(
            path=GPTURK_DATA_DIR / cfg.approach / "train_transductive.json",
            tokenizer=tokenizer,
            train=False
        )
        test = GPTurkDataset(
            path=GPTURK_DATA_DIR / "evaluation.json",
            tokenizer=tokenizer
        )
        augmented_text = GPTurkDataset(
            path=GPTURK_DATA_DIR / cfg.approach / "train_transductive.json",
            tokenizer=tokenizer,
            train=False
        )
    elif cfg.dataset == "gpturk_inductive":
        dataset = GPTurkDataset(
            path=GPTURK_DATA_DIR / cfg.approach / "train_inductive.json",
            tokenizer=tokenizer,
            train=True
        )
        test = GPTurkDataset(
            path=GPTURK_DATA_DIR / "evaluation.json",
            tokenizer=tokenizer
        )
        augmented_text = GPTurkDataset(
            path= GPTURK_DATA_DIR / cfg.approach / "train_inductive.json",
            tokenizer=tokenizer,
            train=True
        )
    elif cfg.dataset == "gpturk":
        dataset = GPTurkDataset(
            path=GPTURK_DATA_DIR / cfg.approach / "train_original.json",
            tokenizer=tokenizer,
            train=True
        )
        test = GPTurkDataset(
            path=GPTURK_DATA_DIR / "evaluation_original.json",
            tokenizer=tokenizer
        )
        augmented_text = GPTurkDataset(
            path= GPTURK_DATA_DIR / cfg.approach / "train_original.json",
            tokenizer=tokenizer,
            train=True
        )
    
    validation_length = cfg.validation_length
    if cfg.use_augmented_data:
        total_train_length = len(augmented_text)
    else:
        total_train_length = len(dataset) 

    indices = list(range(0, total_train_length, 500)) + [total_train_length]

    indices = [idx for idx in indices if idx <= 5000]
    print("indices:", indices)
    
    if not cfg.for_synthetic:
        indices = [len(dataset)-validation_length]
   
    print(indices)
   
    for idx in indices:
        
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"{cfg.ckpt}_size:{idx}",
            group=f"{cfg.dataset}_{cfg.approach}",
            config={
                "ckpt": cfg.ckpt,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "num_epochs": cfg.num_epochs,
                "warmup_ratio": cfg.warmup_ratio,
                "weight_decay": cfg.weight_decay
            }
        )
        print("dataset.num_labels", dataset.num_labels)
        if cfg.multi_label:
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.ckpt, 
                num_labels=dataset.num_labels,
                problem_type="multi_label_classification"
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.ckpt,
                num_labels=dataset.num_labels,
                #hidden_dropout_prob=0.2,
                #attention_probs_dropout_prob=.2
            )

        if cfg.use_augmented_data:
            if idx == 0:
                continue
            train_size = int(idx)
            split_upper = total_train_length - train_size
            train_dataset, left_over = torch.utils.data.random_split(augmented_text, [train_size, split_upper])
            split_upper2 = len(dataset) - validation_length
            eval_dataset, left_over2 = torch.utils.data.random_split(dataset, [validation_length, split_upper2])
        else:
            train_size = int(idx)
            split_upper = total_train_length - train_size
            train_dataset, left_over = torch.utils.data.random_split(dataset, [train_size, split_upper])
            split_upper2 = len(left_over) - validation_length
            eval_dataset, left_over2 = torch.utils.data.random_split(left_over, [validation_length, split_upper2])
            print("len(eval_dataset)", len(eval_dataset))
            print("len(train_dataset)", len(train_dataset))
        
        training_args = TrainingArguments(
                output_dir=MODELS_DIR / f"{cfg.dataset}_{cfg.approach}_size:{idx}_{cfg.ckpt}",
                evaluation_strategy="epoch",
                logging_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                learning_rate=cfg.lr,
                load_best_model_at_end=True,
                per_device_train_batch_size=cfg.batch_size,
                per_device_eval_batch_size=cfg.batch_size,
                num_train_epochs=cfg.num_epochs,
                weight_decay=cfg.weight_decay,
                metric_for_best_model="eval_loss",
                push_to_hub=False,
                seed=42,
                warmup_ratio=cfg.warmup_ratio,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps
            )
        
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics= compute_metrics_multi_label if cfg.multi_label else compute_metrics,
            callbacks=[CustomWandbCallback()],
        )

        trainer.train()   # change label type to float if you want multi label, o/w long
 
        print(trainer.evaluate())
        
        prediction_output = trainer.predict(
            test, metric_key_prefix=""
        )

        metrics, y_pred_logits, y_true = (
            prediction_output.metrics,
            prediction_output.predictions,
            prediction_output.label_ids,
        )
        
        print(y_true)

        trainer.save_model(MODELS_DIR / f"{cfg.dataset}_size:{idx}_{cfg.ckpt}/final")

        # Convert logits to probabilities
        softmax = torch.nn.Softmax()
        probs = softmax(torch.Tensor(y_pred_logits)).numpy()
        if cfg.multi_label:
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs >= 0.3)] = 1
        else:
            y_pred = np.argmax(probs, axis=1)


        clf_report = classification_report(
            y_true=y_true, y_pred=y_pred, target_names=dataset.label_names, output_dict=True
        )
        
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        # Add prefix to metrics "test/"
        metrics = {f"test/{k[1:]}": v for k, v in metrics.items()}
        # Log results
        wandb.log(
            metrics,
        )

        df = pd.DataFrame(clf_report)
        df["metric"] = df.index
        table = wandb.Table(data=df)

        wandb.log(
            {
                "classification_report": table,
            }
        )   
        
        wandb.finish()

if __name__=="__main__":
    main()