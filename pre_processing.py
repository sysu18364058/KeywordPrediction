import pickle
import random
import re

from flashtext import KeywordProcessor
from tqdm import tqdm


def build_keyword_processor(kw_list):
    keyword_processor=KeywordProcessor(case_sensitive=False)
    for kw in kw_list:
        keyword_processor.add_keyword(kw)
    return keyword_processor

def match_keywords(text, keyword_processor):
    return keyword_processor.extract_keywords(text)

def get_labels_num(dic):
    labels = []
    for d in dic:
        for l in d['label']:
            labels.append(l)
    return len(list(set(labels)))

def split_dataset(data, train_ratio, val_ratio):

    random.shuffle(data)
    n_total = len(data)
    offset_1 = int(n_total * train_ratio)
    offset_2 = int(n_total * val_ratio)
    trainset = data[:offset_1]
    valset = data[offset_1:offset_1+offset_2]
    testset = data[offset_1+offset_2:]

    return trainset, valset, testset

raw_dataset = ['2014.txt', '2010.txt']
# raw_dataset = ['temp.txt']

min_turns = 12
min_keywords = 25
min_description_length = 150
keyword_file = ['THUOCL_medical.txt']

dialogues = []

kw_list = []
for kwf in keyword_file:
    f = open(kwf, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        kw_list.append(line.split('\t')[0])

kp = build_keyword_processor(kw_list)

for text in raw_dataset:
    f = open(text, 'r', encoding='utf-8')
    content = f.readlines()
    f.close()
    
    content = ''.join(content)
    content = content.split('id=')[1:]

    for ct in tqdm(content):
        try:
            description = ct.split('Description\n')[1].split('Dialogue')[0].strip('\n')
            description = re.sub('\n', ' ', description)
            description = re.sub('疾病：.*内容：', '', description)
            description = re.sub('疾病：.*病情描述：', '', description)
            description = re.sub('病情描述：', ' ', description)
            description = re.sub('病情描述（发病时间、主要症状、就诊医院等）：', '', description)
            description = re.sub('曾经治疗情况和效果：', '。', description)
            description = re.sub('想得到怎样的帮助:', '。', description)
            if len(description) < min_description_length:
                continue
            dialog = ct.split('Dialogue\n')[1].strip('\n')
        except IndexError:
            pass
        
        sample = {}
        sample['des'] = description
        dialog = re.sub('病人：\n', ' ', dialog)
        dialog = re.sub('医生：\n', ' ', dialog)
        turns = dialog.count('\n') + 2
        if turns < min_turns:
            continue
        key_words = match_keywords(description+dialog, kp)
        if len(list(set(key_words))) > min_keywords:
            sample['sen'] = dialog
            dialogues.append(sample)
wf = open('save.pk', 'wb')
pickle.dump(dialogues,wf)
wf.close()

train, val, test = split_dataset(dialogues, 0.8, 0.1)

wf = open('train.pk', 'wb')
pickle.dump(train, wf)
wf.close()

wf = open('dev.pk', 'wb')
pickle.dump(val, wf)
wf.close()

wf = open('test.pk', 'wb')
pickle.dump(test, wf)
wf.close()
