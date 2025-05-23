from pathlib import Path
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset
import torch
import sys

def translate(sentence: str):
    # Setup device, config, tokenizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()

    tokenizer_src = Tokenizer.from_file(
        str(Path(config['tokenizer_file'].format(config['lang_src'])))
    )
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(config['tokenizer_file'].format(config['lang_tgt'])))
    )

    # Build and load model
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'], config['seq_len'],
        d_model=config['d_model']
    ).to(device)

    ckpt = latest_weights_file_path(config)
    if ckpt is None:
        print("No checkpoint found.")
        return None
    state = torch.load(ckpt, map_location=device)
    # support both state_dict and wrapped dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()

    # If numeric index, fetch from dataset
    label = ""
    if sentence.isdigit():
        idx = int(sentence)
        raw = load_dataset(config['datasource'], f"{config['lang_src']}-{config['lang_tgt']}", split='all')
        ds = BilingualDataset(raw, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])
        src_text = ds[idx]['src_text']
        label = ds[idx]['tgt_text']
    else:
        src_text = sentence

    # Prepare source tensor
    seq_len = config['seq_len']
    enc = tokenizer_src.encode(src_text)
    src_ids = [tokenizer_src.token_to_id('[SOS]')] + enc.ids + [tokenizer_src.token_to_id('[EOS]')]
    src_ids += [tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(src_ids))
    source = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2)

    # Decode step by step
    with torch.no_grad():
        encoder_out = model.encode(source, src_mask)
        decoder_input = torch.full((1,1), tokenizer_tgt.token_to_id('[SOS]'),
                                   dtype=torch.long).to(device)

        print(f"{'SOURCE:':>12} {src_text}")
        if label:
            print(f"{'TARGET:':>12} {label}")
        print(f"{'PREDICTED:':>12} ", end='')

        for _ in range(seq_len-1):
            tgt_mask = torch.triu(
                torch.ones((1, decoder_input.size(1), decoder_input.size(1)), device=device),
                diagonal=1
            ).bool()
            out = model.decode(encoder_out, src_mask, decoder_input, tgt_mask)
            logits = model.project(out[:, -1:])
            next_id = logits.argmax(dim=-1)
            decoder_input = torch.cat([decoder_input, next_id], dim=1)
            token = tokenizer_tgt.decode([next_id.item()])
            print(token, end=' ')
            if next_id.item() == tokenizer_tgt.token_to_id('[EOS]'):
                break
        print()

    # Assemble final translation
    gen_ids = decoder_input.squeeze(0).tolist()[1:]
    eos_id = tokenizer_tgt.token_to_id('[EOS]')
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    translation = tokenizer_tgt.decode(gen_ids, skip_special_tokens=True)
    return translation

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python translate.py <sentence or index>")
        sys.exit(1)
    arg = " ".join(sys.argv[1:])
    result = translate(arg)
    if result:
        print(f"\nFull translation: {result}")