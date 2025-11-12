# AIG2PT/train.py - Training script for AIG generation model
import os
import time
import math
import json
from contextlib import nullcontext
import sys
import logging
import yaml
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, default_data_collator
from pathlib import Path

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("AIG2PT")

SCRIPT_DIR = Path(__file__).parent.absolute()

# Import model
try:
    from core.model import GPT, GPTConfig
    logger.info("Model imported successfully")
except ImportError as e:
    logger.error(f"Failed to import model: {e}")
    sys.exit(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -----------------------------------------------------------------------------
# Dataset for Tokenized AIGs
# -----------------------------------------------------------------------------
class TokenizedAIGDataset(Dataset):
    """Loads pre-tokenized AIG sequences from binary memmap files."""
    def __init__(self, data_dir, pad_token_id=0):
        self.data_dir = Path(data_dir)

        # Load metadata
        meta_path = self.data_dir.parent / 'data_meta.json'
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        split_name = self.data_dir.name
        self.shape = tuple(metadata[split_name]['token_ids_shape'])
        self.num_graphs = metadata[split_name]['num_graphs']
        self.pad_token_id = pad_token_id

        # Load token IDs as memmap
        self.token_ids = np.memmap(
            self.data_dir / 'token_ids.bin',
            dtype=np.int16,
            mode='r',
            shape=self.shape
        )

        logger.info(f"Loaded {split_name}: {self.num_graphs} graphs, max_len={self.shape[1]}")

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        token_ids = torch.from_numpy(self.token_ids[idx].astype(np.int64))
        attention_mask = (token_ids != self.pad_token_id).long()
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': token_ids.clone()
        }


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Learning rate schedule with warmup and cosine decay."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, loader, eval_iters, ctx, device):
    """Estimate loss on dataset."""
    out = []
    model.eval()
    num_eval_iters = 0
    for data in loader:
        if num_eval_iters >= eval_iters:
            break
        X = data['input_ids'][:, :-1].to(device)
        Y = data['labels'][:, 1:].to(device)
        Y_mask = data['attention_mask'][:, 1:].to(device)
        with ctx:
            logits, loss = model(X, Y, Y_mask)
        out.append(loss.item())
        num_eval_iters += 1
    model.train()
    return np.mean(out) if out else float('inf')


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def main():
    # Load configurations from YAML files
    config = {}
    config_files = ['aig.yaml', 'network.yaml', 'train.yaml', 'sample.yaml']
    for fname in config_files:
        path = SCRIPT_DIR / 'configs' / fname
        if path.exists():
            with open(path, 'r') as f:
                config.update(yaml.safe_load(f) or {})

    # Apply overrides from configurator.py if exists
    configurator_path = SCRIPT_DIR / 'configurator.py'
    if configurator_path.exists():
        config_keys = [k for k in config.keys()]
        exec(open(configurator_path).read())
        config.update({k: globals()[k] for k in config_keys if k in globals()})

    # Extract config values
    dataset_name = config.get('name', 'aig')
    out_dir = config.get('out_dir', 'out')
    eval_interval = config.get('eval_interval', 1000)
    log_interval = config.get('log_interval', 10)
    eval_iters = config.get('eval_iters', 200)
    always_save_checkpoint = config.get('always_save_checkpoint', False)
    init_from = config.get('init_from', 'scratch')

    wandb_log = config.get('wandb_log', False)
    wandb_project = config.get('wandb_project', 'aig2pt')
    wandb_run_name = config.get('wandb_run_name', None)

    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 5 * 8)
    batch_size = config.get('batch_size', 12)
    block_size = config.get('block_size', 1024)
    vocab_size = config.get('vocab_size', None)

    n_layer = config.get('n_layer', 12)
    n_head = config.get('n_head', 12)
    n_embd = config.get('n_embd', 768)
    dropout = config.get('dropout', 0.0)
    bias = config.get('bias', False)
    model_name = config.get('model_name', 'base')

    learning_rate = config.get('learning_rate', 1e-4)
    max_iters = config.get('max_iters', 300000)
    weight_decay = config.get('weight_decay', 1e-1)
    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.95)
    grad_clip = config.get('grad_clip', 1.0)

    decay_lr = config.get('decay_lr', True)
    warmup_iters = config.get('warmup_iters', 2000)
    lr_decay_iters = config.get('lr_decay_iters', 300000)
    min_lr = config.get('min_lr', 1e-5)

    backend = config.get('backend', 'nccl')
    device = config.get('device', 'cuda')
    dtype = config.get('dtype', 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')
    compile_model = config.get('compile', False)

    processed_data_dir = config.get('processed_data_dir', str(SCRIPT_DIR / 'dataset' / 'aig_prepared'))
    tokenizer_path = config.get('tokenizer_path', str(SCRIPT_DIR / 'dataset' / 'tokenizer'))

    # Update output directory with run name
    if wandb_log and wandb_run_name is None:
        wandb_run_name = f"{dataset_name}-{model_name}"
    out_dir = f'results/{wandb_run_name}' if wandb_run_name else out_dir

    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    if 'cuda' in device:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Load datasets
    train_dataset = TokenizedAIGDataset(os.path.join(processed_data_dir, 'train'), pad_token_id)
    eval_dataset = TokenizedAIGDataset(os.path.join(processed_data_dir, 'val'), pad_token_id)

    # Data collator
    def data_collate_fn(features):
        features = default_data_collator(features)
        seq_len = features['attention_mask'].sum(-1)
        max_len = seq_len.max()
        features = {k: v[..., :max_len] for k, v in features.items()}
        return features

    # Data loaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True,
        drop_last=False,
        num_workers=8,
        collate_fn=data_collate_fn
    )

    eval_sampler = DistributedSampler(eval_dataset) if ddp else None
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=8,
        collate_fn=data_collate_fn
    )

    # Initialize model
    iter_num = 0
    best_val_loss = 1e9

    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout
    )

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size

    model.to(device)

    # Gradient scaler
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

    # Optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    # Compile model
    if compile_model:
        print("compiling the model...")
        model = torch.compile(model)

    # Wrap in DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Wandb logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # Training loop
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0

    micro_step = 0
    while True:
        for data in train_loader:
            X = data['input_ids'][:, :-1].to(device)
            Y = data['labels'][:, 1:].to(device)
            Y_mask = data['attention_mask'][:, 1:].to(device)

            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

            with ctx:
                logits, loss = model(X, Y, Y_mask)
                loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            micro_step += 1

            if micro_step == gradient_accumulation_steps:
                micro_step = 0
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                if iter_num % log_interval == 0 and master_process:
                    lossf = loss.item() * gradient_accumulation_steps
                    if local_iter_num >= 5:
                        mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                        running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

                iter_num += 1
                local_iter_num += 1

                if iter_num > max_iters:
                    break

                # Learning rate schedule
                lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Evaluation
                if iter_num % eval_interval == 0 and master_process and iter_num != 0:
                    train_loss = estimate_loss(raw_model, train_loader, eval_iters, ctx, device)
                    val_loss = estimate_loss(raw_model, eval_loader, eval_iters, ctx, device)
                    print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                    if wandb_log:
                        wandb.log({
                            "iter": iter_num,
                            "train/loss": train_loss,
                            "val/loss": val_loss,
                            "lr": lr,
                            "mfu": running_mfu * 100,
                        })
                    if val_loss < best_val_loss or always_save_checkpoint:
                        best_val_loss = val_loss
                        if iter_num > 0:
                            checkpoint = {
                                'model': raw_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'model_args': model_args,
                                'iter_num': iter_num,
                                'best_val_loss': best_val_loss,
                                'config': config,
                            }
                            print(f"saving checkpoint to {out_dir}")
                            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()

