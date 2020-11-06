from torch.nn.utils.rnn import pad_sequence
import random
from abc import ABC, abstractmethod
from src.tokenization_utils import PreTrainedTokenizer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Tuple

class DataCollator(ABC):
    """
    A `DataCollator` is responsible for batching
    and pre-processing samples of data as requested by the training loop.
    """

    @abstractmethod
    def collate_batch(self) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.

        Returns:
            A dictionary of tensors
        """
        pass

@dataclass
class PredictDatacollator(DataCollator):

    def __init__(self, 
                 tokenizer: PreTrainedTokenizer = None,
                 mlm_prob: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        
    def collate_batch(self, 
                     examples: list):
        all_sens = []   # List[torch.Tensor]
        labels = []
        for ex in examples:
            sen, label = self.mask_keyword_tokens(ex)
            all_sens.append(sen)
            labels.append(label)
        length_of_first = all_sens[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in all_sens)
        if are_tensors_same_length:
            batch = torch.stack(all_sens, dim=0)
            labels = torch.stack(labels, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            batch = pad_sequence(all_sens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return {"input_ids": batch, "labels": labels}
    
    def mask_keyword_tokens(self,
                    inputs: dict,):
        labels = torch.tensor(inputs['sen'], dtype=torch.long)
        # get keyword_tokens in sentences
        for keyword in inputs['keywords']:
            keyword_index =  [i for i in range(len(inputs['sen'])) if inputs['sen'][i] == keyword]
            for i in keyword_index:
                if random.random() < self.mlm_prob:
                    labels[~i] = -100  # only compute loss on masked tokens
                    prob = random.random()
                    if prob < 0.8:  # 80% mask
                        inputs['sen'][i] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                    elif prob < 0.9:  # 10% replace with random word
                        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
                        inputs['sen'][i] = random_words[i]
                    # 10% keep unchanged
                    
        return torch.tensor(inputs['sen']), labels