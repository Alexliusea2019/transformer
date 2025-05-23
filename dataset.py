import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds  # ✅ this line will now work
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]

        # Tokenize source and target
        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Truncate
        src_tokens = src_tokens[:self.seq_len]
        tgt_tokens = tgt_tokens[:self.seq_len - 2]  # for [SOS] and [EOS]

        # Add [SOS] and [EOS]
        decoder_input_tokens = [self.sos_token.item()] + tgt_tokens
        decoder_output_tokens = tgt_tokens + [self.eos_token.item()]

        # Pad
        src_tokens += [self.pad_token.item()] * (self.seq_len - len(src_tokens))
        decoder_input_tokens += [self.pad_token.item()] * (self.seq_len - len(decoder_input_tokens))
        decoder_output_tokens += [self.pad_token.item()] * (self.seq_len - len(decoder_output_tokens))

        # Convert to tensors
        encoder_input = torch.tensor(src_tokens, dtype=torch.long)
        decoder_input = torch.tensor(decoder_input_tokens, dtype=torch.long)
        target = torch.tensor(decoder_output_tokens, dtype=torch.long)

        # Masks
        encoder_mask = (encoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0)
        decoder_mask = causal_mask(decoder_input.size(0))

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "target": target,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size: int):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def collate_fn(batch):
    encoder_input = torch.stack([item["encoder_input"] for item in batch])
    decoder_input = torch.stack([item["decoder_input"] for item in batch])
    target = torch.stack([item["target"] for item in batch])

    src_mask = (encoder_input != batch[0]["encoder_input"].new_full((1,), batch[0]["encoder_input"].min())).unsqueeze(1).unsqueeze(2)
    tgt_mask = (decoder_input != batch[0]["decoder_input"].new_full((1,), batch[0]["decoder_input"].min())).unsqueeze(1).unsqueeze(2)
    size = decoder_input.size(1)
    causal = causal_mask(size).to(decoder_input.device)
    combined_mask = tgt_mask & causal

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "target": target,
        "src_mask": src_mask,
        "tgt_mask": combined_mask,
    }