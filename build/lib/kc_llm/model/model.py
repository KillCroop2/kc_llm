import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_positions=1024, n_embd=768, n_layer=12, n_head=12, dropout=0.1):
        super(GPTModel, self).__init__()
        configuration = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False
        )
        self.model = GPT2LMHeadModel(configuration)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, start_text, tokenizer, max_length=200, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95):
        input_ids = tokenizer.encode(start_text, return_tensors='pt').to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]