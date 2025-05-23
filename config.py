from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

from pathlib import Path
from tqdm import tqdm
import os
import wandb
import torchmetrics

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

# Greedy decoder for validation
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64).unsqueeze(0).to(device)
    eos_token_id = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = sos_token

    for _ in range(max_len):
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        out = model.project(out)
        next_token = out.argmax(dim=-1)[:, -1].unsqueeze(1)
        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        if next_token.item() == eos_token_id:
            break

    return decoder_input.squeeze(0)

# Run validation step
def run_validation(model, val_loader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    actuals, predictions = [], []

    for batch in val_loader:
        encoder_input = batch['encoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)

        output_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
        output_text = tokenizer_tgt.decode(output_tokens.tolist(), skip_special_tokens=True)

        target_text = tokenizer_tgt.decode(batch['target'].squeeze(0).tolist(), skip_special_tokens=True)
        predictions.append(output_text)
        actuals.append(target_text)

        if num_examples > 0:
            print_msg(f"{target_text} -> {output_text}")
            num_examples -= 1

    bleu = torchmetrics.functional.bleu_score(
        [[p.split()] for p in predictions],
        [[a.split()] for a in actuals],
        n_gram=4
    )

    wandb.log({'validation/BLEU': bleu.item(), 'global_step': global_step})
    model.train()

# Tokenizer utility
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)

    tokenizer.save(str(tokenizer_path))
    return tokenizer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_ds(config):
    ds_raw = load_dataset(config['datasource'], f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    ds_train_raw, ds_val_raw = random_split(ds_raw, [train_size, val_size])

    train_ds = BilingualDataset(ds_train_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(ds_val_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(
        vocab_src_len, vocab_tgt_len,
        config["seq_len"], config['seq_len'],
        d_model=config['d_model']
    )

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"))

    global_step = 0
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        loop = tqdm(train_loader, leave=True)
        total_loss = 0

        for batch in loop:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            target = batch["target"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            logits = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            loop.set_description(f"Epoch [{epoch}/{config['num_epochs']}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        wandb.log({"train/loss": avg_loss, "epoch": epoch, "global_step": global_step})

        run_validation(model, val_loader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, print, global_step)

        model_path = get_weights_file_path(config, str(epoch))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    config = get_config()
    config['num_epochs'] = 30
    config['preload'] = None

    wandb.init(
        project="pytorch-transformer",
        config=config
    )

    train_model(config)
