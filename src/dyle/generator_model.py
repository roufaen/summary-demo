import torch
import numpy as np

from model_center.model import CPM1


class GeneratorModel(torch.nn.Module):

    def __init__(self, cpm_path, top_k, max_source_len, max_target_len):
        super().__init__()
        self.generator_model = CPM1.from_pretrained(cpm_path)
        generator_dim = 1024
        self.dynamic_score_projection = torch.nn.Sequential(
            torch.nn.Linear(in_features=generator_dim, out_features=generator_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=generator_dim, out_features=generator_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=generator_dim, out_features=1),
        ).cuda()
        self.top_k = top_k
        self.max_source_len = max_source_len

    def forward(self, input_tokens, input_length, input_context, input_span, past_key_values=None):
        batch_size = input_tokens.shape[0]
        target_len = input_tokens.shape[1] - self.top_k * self.max_source_len
        input_tokens_ = input_tokens[:, :self.top_k * self.max_source_len].contiguous().view(batch_size * self.top_k, self.max_source_len)
        input_length_ = torch.full((batch_size * self.top_k,), self.max_source_len + target_len).cuda().int()
        input_context_ = torch.cat((torch.ones((batch_size * self.top_k, self.max_source_len)).cuda(), input_context[:, self.top_k * self.max_source_len:].repeat(1, self.top_k).view(batch_size * self.top_k, target_len)), dim=1).cuda().int()
        input_span_ = torch.ones_like(input_tokens_).cuda().int()
        for i in range(batch_size * self.top_k):
            input_span_[i, 0:self.max_source_len - len(torch.nonzero(input_tokens_[i]))] = torch.zeros((self.max_source_len - len(torch.nonzero(input_tokens_[i])),)).cuda()
        input_tokens_ = torch.cat((input_tokens_, input_tokens[:, self.top_k * self.max_source_len:].repeat(1, self.top_k).view(batch_size * self.top_k, target_len)), dim=1).cuda().int()
        input_span_ = torch.cat((input_span_, torch.ones((batch_size * self.top_k, target_len)).cuda()), dim=1).cuda().int()

        if past_key_values is None:
            logits, hidden_states, present_key_values = self.generator_model(input_tokens_, input_length_, input_context_, input_span_)
        else:
            for i in range(len(past_key_values)):
                past_key_values[i][0] = torch.cat(torch.split(past_key_values[i][0], 1), dim=1).squeeze(0)
                past_key_values[i][1] = torch.cat(torch.split(past_key_values[i][1], 1), dim=1).squeeze(0)
            logits, hidden_states, present_key_values = self.generator_model(input_tokens_[:, -1:], input_length_, input_context_, input_span_, past_key_values)
            logits = torch.cat((self.logits, logits), dim=1)
            hidden_states = torch.cat((self.hidden_states, hidden_states), dim=1)
        for i in range(len(present_key_values)):
            present_key_values[i][0] = torch.stack(torch.split(present_key_values[i][0], self.top_k), dim=0)
            present_key_values[i][1] = torch.stack(torch.split(present_key_values[i][1], self.top_k), dim=0)
        self.logits = logits
        self.hidden_states = hidden_states

        logits = logits.float()  # (batch_size * top_k, max_source_len + max_target_len, vocab_size)
        dynamic_scores = self.dynamic_score_projection(hidden_states.float()).squeeze(2)  # (batch_size * top_k, max_source_len + max_target_len)

        logits = logits[:, self.max_source_len - 1:].contiguous().view(batch_size, self.top_k, target_len + 1, -1)  # (batch_size, top_k, max_target_len, vocab_size)
        dynamic_scores = dynamic_scores[:, self.max_source_len - 1:].view(batch_size, self.top_k, target_len + 1)  # (batch_size, top_k, max_target_len)

        seq_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)  # (batch_size, top_k, max_target_len, vocab_size)
        doc_logprobs = torch.log_softmax(dynamic_scores, dim=1)  # (batch_size, top_k, max_target_len)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1)  # (batch_size, top_k, max_target_len, vocab_size)
        logprobs = torch.logsumexp(log_prob_sum, dim=1)  # (batch_size, max_target_len, vocab_size)

        return logprobs, dynamic_scores, present_key_values
