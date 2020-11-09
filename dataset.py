import pickle

from flashtext import KeywordProcessor

from torch.utils.data.dataset import Dataset

from src.tokenization_utils import PreTrainedTokenizerBase


class TextDataset(Dataset):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerBase,
                 file_path: str,
                 block_size: int,
                 max_tokens: int,
                 keyword_file: str):
        kw_list = []
        f = open(keyword_file, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            kw_list.append(line.split('\t')[0])
        self.keyword_processor = self.build_keyword_processor(kw_list)

        self.examples = []
        with open(file_path, 'rb') as dataset:
            ds = pickle.load(dataset)
            for instance in ds:
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instance['sen']))
                dialogue = instance['sen'].split('\n') # ['sen1', 'sent2', ...]
                for i in range(0, len(dialogue)-block_size+1, block_size):
                    block = dialogue[i:i+block_size]
                    keywords = self.match_keywords(''.join(block), self.keyword_processor)
                    # keywords = ''.join(keywords)
                    # keywords = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(keywords))
                    example = {}
                    example['sen'] = []
                    for j, sen in enumerate(block):
                        block[j] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen))
                        example['sen'] += tokenizer.build_inputs_with_special_tokens(block[j])
                    example['sen'] = example['sen'][:max_tokens]
                    # example['keywords'] = tokenizer.build_inputs_with_special_tokens(keywords)
                    example['keywords'] = keywords
                    self.examples.append(example)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> dict: # {'sen':[sentences_token_ids], 'keywords':[keywords_token_ids]}
        return self.examples[idx]

    def build_keyword_processor(self, kw_list):
        keyword_processor=KeywordProcessor(case_sensitive=False)
        for kw in kw_list:
            keyword_processor.add_keyword(kw)
        return keyword_processor

    def match_keywords(self, text, keyword_processor):
        return keyword_processor.extract_keywords(text)

