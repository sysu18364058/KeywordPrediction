# KeywordPrediction
As a pretrain task for a bert model, predict masked keywords in medical dialogues

use whole word mask.
e.g. ['我', '喜', '欢', '你']  label:['喜', '##欢'],  masked sentence: ['我', '[MASK]', '[MASK]'， '你']

TODO:
use some metrics to evaluate our model on the test dataset 
