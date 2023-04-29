from tqdm import tqdm
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import math
import wandb
from transformers import get_scheduler

from ED.utils import *
from ED.models import *
from common_utils.common_ML_utils import *
from common_utils.common_data_processing_utils import *
from ED.data_loaders import load_bi_encoder_input, padding_bi_encoder_data, load_triplet_encoder_input
from CTC.TE_loaders import load_DataWithTokenType

def train_cross_encoder_notebook(config):
    model = CrossEocoder(config)
    df = pd.read_pickle(config.input_file)
    if getattr(config, 'drop_ones_with_cell_ref', False):
        print('dropping ones with cell_reference')
        print(len(df))
        df = df[df.cell_reference=='']
        print(len(df))
    g = set_seed(config.seed)
    input_num_cols = [] if not hasattr(config, 'input_num_cols') else config.input_num_cols
    train_ds, valid_ds, _ = load_DataWithTokenType(config, df, config.input_cols, input_num_cols, config.valid_fold, config.test_fold, augment=False)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=config.BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=g)
    eval_dl = DataLoader(valid_ds, batch_size=config.eval_BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=g)
    EL_train_loop_notebook(config, model, train_dl, eval_dl)


def cross_encoder(g, args, config, eval_steps=150):
    """
    :param val_test_fold_file: if not None, use the same fold for validation and testing. The split is determined by the file.
    """

    model = CrossEocoder(config)

    df = pd.read_pickle(args.input_file)

    if getattr(config, 'drop_ones_with_cell_ref', False):
        print('dropping ones with cell_reference')
        print(len(df))
        df = df[df.cell_reference=='']
        print(len(df))
    
    # if args.val_test_fold_file is None:
    train_ds, valid_ds, _ = load_DataWithTokenType(config, df, config.input_cols, config.input_num_cols, 'img_class', args.test_fold, augment=False)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=g)
    eval_dl = DataLoader(valid_ds, batch_size=256, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=g)
    # else:
    #     with open(args.val_test_fold_file) as f:
    #         split = json.load(f)
    #     train_df = df[df.fold != args.test_fold].copy()
    #     valid_df = df[(df.fold == args.test_fold) & (df.paper_id.isin(split['val'][args.test_fold]))].copy()
    #     train_ds = load_single_DataWithTokenType(config, train_df, drop_duplicates=True)
    #     valid_ds = load_single_DataWithTokenType(config, valid_df, drop_duplicates=False)
    #     train_dl = DataLoader(train_ds, shuffle=True, batch_size=config.BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=g)
    #     eval_dl = DataLoader(valid_ds, batch_size=256, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=g)

    EL_train_loop(args, model, train_dl, eval_dl, eval_steps)


def bi_encoder_triplet(g, args, config, eval_steps=150, valid_fold='img_class'):
    """
    Train a bi-encoder model with triplet loss for direct entity search.
    """
    DATA_ROOT_DIR = Path(args.data_root_dir)

    model = BiEncoderTriplet(config)

    df = pd.read_pickle(DATA_ROOT_DIR / config.input_file)

    if getattr(config, 'drop_ones_with_cell_ref', False):
        print('dropping ones with cell_reference')
        print(len(df))
        df = df[df.cell_reference=='']
        print(len(df))

    train_df, valid_df, _ = split_fold(df, valid_fold, args.test_fold)
    train_ds, valid_ds = load_triplet_encoder_input(config, train_df, valid_df)

    print(f"Training length {len(train_ds)}")
    print(f"Validation length {len(valid_ds)}")

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=config.BS, collate_fn=padding_bi_encoder_data, worker_init_fn=seed_worker, generator=g)
    eval_dl = DataLoader(valid_ds, batch_size=64, collate_fn=padding_bi_encoder_data, worker_init_fn=seed_worker, generator=g)

    EL_train_loop(args, model, train_dl, eval_dl, eval_steps)


def bi_encoder(g, args, config, eval_steps=150):
    DATA_ROOT_DIR = Path(args.data_root_dir)

    model = BiEncoder(config)

    df = pd.read_pickle(DATA_ROOT_DIR / config.input_file)

    if getattr(config, 'drop_ones_with_cell_ref', False):
        print('dropping ones with cell_reference')
        print(len(df))
        df = df[df.cell_reference=='']
        print(len(df))

    train_df, valid_df, _ = split_fold(df, 'img_class', args.test_fold)
    train_ds, valid_ds = load_bi_encoder_input(config, train_df, valid_df)

    print(f"Training length {len(train_ds)}")
    print(f"Validation length {len(valid_ds)}")

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=config.BS, collate_fn=padding_bi_encoder_data, worker_init_fn=seed_worker, generator=g)
    eval_dl = DataLoader(valid_ds, batch_size=64, collate_fn=padding_bi_encoder_data, worker_init_fn=seed_worker, generator=g)

    EL_train_loop(args, model, train_dl, eval_dl, eval_steps)


def EL_train_loop_notebook(config, model, train_dl, eval_dl):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    num_training_steps = math.ceil(config.epoch * len(train_dl) / config.grad_accum_step)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )
    step_count = 0
    min_val_loss = None
    accum_loss = 0
    pbar = tqdm(total=num_training_steps)
    for _ in range(config.epoch):
        for batch_idx, batch in enumerate(train_dl):
            model.train()
            outputs = model(**batch)
            loss = outputs.loss / config.grad_accum_step
            accum_loss += loss.detach()

            loss.backward()

            if ((batch_idx + 1) % config.grad_accum_step == 0) or (batch_idx + 1 == len(train_dl)):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                accum_loss = 0
                pbar.update(1)

                # Validation
                if step_count % config.eval_steps == 0:                    
                    val_loss, _ = generate_relavance_score(model, eval_dl)
                    if min_val_loss is None or val_loss < min_val_loss:
                        min_val_loss = val_loss
                        if config.save_dir is not None:
                            os.makedirs(config.save_dir, exist_ok=True)
                            torch.save(model, os.path.join(config.save_dir, config.name))
    pbar.close()


def EL_train_loop(args, model, train_dl, eval_dl, eval_steps):
    """The training loop for all models"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    num_training_steps = math.ceil(args.epoch * len(train_dl) / args.grad_accum_step)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    step_count = 0
    min_val_loss = None
    accum_loss = 0
    pbar = tqdm(total=num_training_steps)
    for _ in range(args.epoch):
        for batch_idx, batch in enumerate(train_dl):
            model.train()
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_step
            accum_loss += loss.detach()

            loss.backward()

            if ((batch_idx + 1) % args.grad_accum_step == 0) or (batch_idx + 1 == len(train_dl)):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                if args.use_wandb:
                    wandb.log({"training_loss": accum_loss}, step=step_count)
                accum_loss = 0
                pbar.update(1)

                # Validation
                if step_count % eval_steps == 0:                    
                    val_loss, _ = generate_relavance_score(model, eval_dl)

                    if args.use_wandb:
                        wandb.log({'val_loss': val_loss}, step=step_count)

                    if min_val_loss is None or val_loss < min_val_loss:
                        min_val_loss = val_loss
                        if args.use_wandb:
                            wandb.run.summary[f"best_val_loss"] = min_val_loss
                        if args.model_dir is not None:
                            torch.save(model, os.path.join(args.model_dir, args.project, args.name))
    pbar.close()