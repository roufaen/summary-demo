import json

from tqdm import tqdm
from typing import Tuple


class InferDataset:
	def __init__(self):
		self.__idx = 0
		self._data = []

	def __len__(self):
		return len(self._data)

	def __iter__(self):
		self.__idx = 0

	def __next__(self):
		if self.__idx < len(self._data):
			data = self._data[self.__idx]
			self.__idx += 1
			return data['text'], data['summary']
		else:
			raise StopIteration

	def __getitem__(self, key) -> Tuple[str, str]:
		if key >= len(self._data):
			return self._data[0]['text'], self._data[0]['summary']
		return	self._data[key]['text'], self._data[key]['summary']


class LCSTSInferDataset(InferDataset):
	def __init__(self, file_path):
		super().__init__()
		with open(file_path, 'r') as f:
			for line in tqdm(f):
				line_data = json.loads(line)
				summary = line_data['summary']
				text = line_data['text']
				self._data.append({'summary': summary, 'text': text})


class CNewSumInferDataset(InferDataset):
	def __init__(self, file_path, max_length=1024):
		super().__init__()
		with open(file_path, 'r') as f:
			for line in tqdm(f):
				line_data = json.loads(line)
				text = ''.join(line_data['article']).replace(' ', '')[:max_length]
				summary = line_data['summary'].replace(' ', '')
				self._data.append({'text':text, 'summary':summary})


INFER_DATASET = {
	'LCSTS': LCSTSInferDataset,
	'CNewSum': CNewSumInferDataset
}
