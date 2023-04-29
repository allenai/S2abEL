import torch
from torch.nn import CrossEntropyLoss
from pathlib import Path
import matplotlib.pyplot as plt
# from sklearn import metrics as skmetrics
from torch.utils.data import DataLoader
import wandb
from transformers import DataCollatorWithPadding, AutoTokenizer
import pandas as pd
from transformers import get_scheduler
import operator
import numpy as np

from CTC.utils import *
from CTC.TE_loaders import *
from CTC.models import *
from common_utils.common_data_processing_utils import split_fold



def test_model(model, test_dl, metrics, cm=False, get_loss=False):
    """
    Run the model on both validation and test folds to produce classfication reports.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    y_preds = []
    y_trues = []
    losses = []
    loss_func = CrossEntropyLoss(reduction='none')

    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        labels = batch["labels"].view(-1)
        idx = (labels != -100).nonzero(as_tuple=True)[0]
        for metric in metrics:
            metric.instance.add_batch(predictions=predictions[idx], references=labels[idx])

        if cm:
            y_preds.append(predictions[idx].cpu())
            y_trues.append(labels[idx].cpu())
        if get_loss:
            with torch.no_grad():
                loss = loss_func(logits.view(-1, 5), batch["labels"].view(-1))
                losses.append(loss.cpu())
    
    if cm:
        y_preds = torch.cat(y_preds, dim=0).squeeze()
        y_trues = torch.cat(y_trues, dim=0).squeeze()
    if get_loss:
        losses = torch.cat(losses, dim=0).squeeze()

    rst = {metric.prefix+metric.name: metric.instance.compute(**metric.keywargs)[metric.name] for metric in metrics}

    for k, v in rst.items():
        if isinstance(v, np.ndarray):
            rst[k] = v.round(4)
        else:
            try:
                rst[k] = round(v, 4)
            except:
                pass
            
    if get_loss:
        return {'metrics': rst, 'cm': (y_trues, y_preds), 'losses': losses}
    else:
        return {'metrics': rst, 'cm': (y_trues, y_preds)}


def train_CTC_notebook(config):
    model = SciBertWithAdditionalFeatures(config)
    df = pd.read_pickle(config.input_file)
    g = set_seed(config.seed)
    input_num_cols = [] if not hasattr(config, 'input_num_cols') else config.input_num_cols
    train_ds, valid_ds, _ = load_DataWithTokenType(config, df, config.input_cols, input_num_cols, config.valid_fold, config.test_fold, config.augment)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=config.BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=g)
    eval_dl = DataLoader(valid_ds, batch_size=config.eval_BS, collate_fn=padding_by_batch)
    CTC_train_loop_notebook(config, model, train_dl, eval_dl, config.name)


def train_CTC(g, args, config, eval_steps=150):

    model = SciBertWithAdditionalFeatures(config)

    df = pd.read_pickle(args.input_file)
    train_ds, valid_ds, test_ds = load_DataWithTokenType(config, df, config.input_cols, config.input_num_cols, args.valid_fold, args.test_fold, config.augment)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=config.BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=g)
    eval_dl = DataLoader(valid_ds, batch_size=512, collate_fn=padding_by_batch)

    CTC_train_loop(args, config, model, train_dl, eval_dl, eval_steps=eval_steps)

    if args.save_dir is not None:
        del model
        model =  torch.load(os.path.join(args.save_dir, args.project, f"{args.name}"))
    test_dl = DataLoader(test_ds, batch_size=512, collate_fn=padding_by_batch)

    rst = test_model(model, test_dl, metrics, cm=True)
    # confusion_matrix = skmetrics.confusion_matrix(*rst['cm'])
    # cr = classification_report(*rst['cm'])
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, 
                        # display_labels=['Other', 'Dataset', 'Method', 'Metric', 'DatasetAndMetric'])
    
    if args.use_wandb:
        # try:
        #     disp.plot()
        #     del rst['cm']
        #     wandb.log({"test cm": plt})
        # except:
        #     pass

        # wandb.run.summary['test_CR'] = cr

        for k, v in rst.items():
            if not isinstance(v, np.ndarray):
                wandb.run.summary['test_' + k] = v
                print(f"test_{k} = {v}")
            else:
                report = rst['metrics']['f1']
                wandb.run.summary['test_Dataset_f1'] = report[1]
                wandb.run.summary['test_DatasetAndMetric_f1'] = report[4]
                wandb.run.summary['test_Method_f1'] = report[2]
                wandb.run.summary['test_Metric_f1'] = report[3]
                wandb.run.summary['test_Other_f1'] = report[0]


def CTC_train_loop_notebook(config, model, train_dl, eval_dl, saved_model_name):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model.to(device)
    num_training_steps = config.num_epochs * len(train_dl)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    step_count = 0
    min_val_loss = None
    bar = tqdm(total=num_training_steps)
    for _ in range(config.num_epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step_count += 1
            bar.update(1)

             # Validation
            if step_count % config.eval_steps == 0:
                model.eval()
                total_loss = 0
                for batch in eval_dl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    total_loss += outputs.loss

                    predictions = torch.argmax(outputs.logits, dim=-1)

                    labels = batch["labels"].view(-1)
                    idx = (labels != -100).nonzero(as_tuple=True)[0]

                    for metric in val_metrics:
                        metric.instance.add_batch(predictions=predictions[idx], references=labels[idx])

                report = {metric.prefix+metric.name: metric.instance.compute(**metric.keywargs)[metric.name] for metric in val_metrics}
                report['val_loss'] = total_loss / len(eval_dl)
                if min_val_loss is None or report['val_loss'] < min_val_loss:
                    min_val_loss = report['val_loss']
                    # print(f'new best loss: {min_val_loss}')
                    if config.save_dir is not None:
                        os.makedirs(config.save_dir, exist_ok=True)
                        torch.save(model, os.path.join(config.save_dir, saved_model_name))
    bar.close()


def CTC_train_loop(args, config, model, train_dl, eval_dl, eval_steps=150):
    """The training loop for all models"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    num_training_steps = config.num_epochs * len(train_dl)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    step_count = 0
    min_val_loss = None
    bar = tqdm(total=num_training_steps)
    for _ in range(config.num_epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            if args.use_wandb:
                wandb.log({"training_loss": loss})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step_count += 1
            bar.update(1)
        
            # Validation
            if step_count % eval_steps == 0:
                model.eval()
                total_loss = 0
                for batch in eval_dl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    total_loss += outputs.loss

                    predictions = torch.argmax(outputs.logits, dim=-1)

                    labels = batch["labels"].view(-1)
                    idx = (labels != -100).nonzero(as_tuple=True)[0]

                    for metric in val_metrics:
                        metric.instance.add_batch(predictions=predictions[idx], references=labels[idx])

                report = {metric.prefix+metric.name: metric.instance.compute(**metric.keywargs)[metric.name] for metric in val_metrics}
                report['val_loss'] = total_loss / len(eval_dl)
                if args.use_wandb:
                    wandb_log(report, step_count=step_count)

                if min_val_loss is None or \
                    report['val_loss'] < min_val_loss:
                    # report['val_micro_f1'] > wandb.run.summary['val_f1_at_best_val_loss']:
                # (report['val_loss'] < min_val_loss and report['val_micro_f1'] > wandb.run.summary['val_f1_at_best_val_loss']):
                    min_val_loss = report['val_loss']
                    if args.use_wandb:
                        wandb.run.summary[f"best_val_loss"] = report['val_loss']
                        wandb.run.summary['val_f1_at_best_val_loss'] = report['val_micro_f1']
                        wandb.run.summary['val_best_Dataset_f1'] = report['val_f1'][1]
                        wandb.run.summary['val_best_DatasetAndMetric_f1'] = report['val_f1'][4]
                        wandb.run.summary['val_best_Method_f1'] = report['val_f1'][2]
                        wandb.run.summary['val_best_Metric_f1'] = report['val_f1'][3]
                        wandb.run.summary['val_best_Other_f1'] = report['val_f1'][0]
                        wandb.run.summary['best_step'] = step_count

                    if args.save_dir is not None:
                        torch.save(model, os.path.join(args.save_dir, args.project, args.name))
    bar.close()
