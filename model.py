import argparse
import random
import math
import os
import torch
from flashtext import KeywordProcessor

from dataset import TextDataset
from src.data.data_collator import DataCollator, WWMDataCollator
from src.configuration_bert import BertConfig
from src.modeling_auto import AutoModelWithLMHead
from src.tokenization_auto import AutoTokenizer
from src.tokenization_utils import PreTrainedTokenizer
from src.trainer import Trainer
from src.training_args import TrainingArguments
import logging

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='bert')
parser.add_argument('--train_data_file', default='train.pk',
                    help='The input training data file (a text file).')
parser.add_argument('--eval_data_file', default='dev.pk',
                    help='An optional input evaluation data file to evaluate the perplexity on (a text file).')
parser.add_argument('--tokenizer', default='bert-base-chinese', 
                    help='Pretrained tokenizer')
parser.add_argument('--keyword_file', default='THUOCL_medical.txt',
                    help='Keyword file, each line starts with a keyword')
parser.add_argument('--block_size', type=int, default=5, 
                    help='Input sequence length after tokenization')
parser.add_argument('--output_dir', default='./runs',
                    help='The output directory where the model predictions and checkpoints will be written.')
args = parser.parse_args()

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = BertConfig()
    model = AutoModelWithLMHead.from_config(config)
    model.resize_token_embeddings(len(tokenizer))
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=args.train_data_file, max_tokens=512,
                                block_size=args.block_size, keyword_file=args.keyword_file)
    eval_dataset = TextDataset(tokenizer=tokenizer, file_path=args.eval_data_file, max_tokens=512,
                                block_size=args.block_size, keyword_file=args.keyword_file)
    data_collator = WWMDataCollator(tokenizer=tokenizer)
    training_args = TrainingArguments(output_dir=args.output_dir, per_device_train_batch_size=8,  
                                      per_device_eval_batch_size=8, num_train_epochs=3)

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, prediction_loss_only=True,
                      train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer)
    # Training
    trainer.train(model_path=None)
    trainer.save_model()

    # Evaluation
    results = {}
    logger.info("*** Evaluate ***")

    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    results.update(result)

if __name__ == '__main__':
    main(args)
