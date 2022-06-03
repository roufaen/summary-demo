import torch

import torch.nn as nn
import torch.nn.functional as F


class CrossSegBert(torch.nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model
		self.classifier = nn.Linear(768, 2, bias=False)

	def forward(self, **model_input):
		result = self.model(**model_input)
		logits = self.classifier(result.last_hidden_state[:, 0].contiguous())
		return logits
