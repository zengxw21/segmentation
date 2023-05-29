
import torch
from transformers  import BertTokenizer
from transformers import AutoModel
from sklearn.model_selection import  train_test_split
from torch.utils.data import TensorDataset,DataLoader

import numpy as np  
from torch import nn   
from transformers import AdamW,BertTokenizer,BertModel
import torch.nn.functional as F     
from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from read_data import GetCorpus,Word2Id
import re

with open("Dataset/train.txt",encoding='utf-8',errors="ignore") as f:
    sentences=f.readlines()
new_list=[]
for each in sentences:
    each=each[1:]
    each=re.sub('\s|\t|\n',"",each)
    new_list.append(each)

with open("Dataset/test.txt",encoding='utf-8',errors="ignore") as f:
    sentences=f.readlines()
test_list=[]
for each in sentences:
    each=each[1:]
    each=re.sub('\s|\t|\n',"",each)
    test_list.append(each)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
max_len = 128
temp_ids = tokenizer(new_list,padding=True,truncation=True,max_length=max_len)
input_ids=temp_ids['input_ids']
attention_mask=temp_ids['attention_mask']

test_ids=tokenizer(new_list,padding=True,truncation=True,max_length=max_len)
test_input_ids=test_ids['input_ids']
test_attention_mask=test_ids['attention_mask']

train_data=torch.tensor(input_ids)
train_masks=torch.tensor(attention_mask)
train_label=torch.tensor()
test_data=torch.tensor(test_input_ids)
test_masks=torch.tensor(test_attention_mask)


batch_size=16

train_dataset=TensorDataset(train_data,train_masks)
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataset=TensorDataset(test_data,test_masks)
test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

bert=AutoModel.from_pretrained('bert-base-chinese')
class BERT(nn.Module):
    def __init__(self,bert):
        super().__init__()
        self.bert=bert
        self.fc1=nn.Linear(768,2)
        def forward(self,sent_id,mask):
            _,cls_hs=self.bert(sent_id,attention_mask = mask,return_dict=False)
            return F.log_softmax(self.fc1(cls_hs),dim=1)
model = BERT(bert)
model = model.cuda()
# 优化算法
optimizer = AdamW(model.parameters(),lr = 1e-5)
# 损失函数
criterion = nn.NLLLoss()
epochs = 10
for e in range(epochs):
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        # 上传至GPU上进行训练
        batch = [i.cuda() for i in batch]
        sent_id,masks,labels = batch
        
        optimizer.zero_grad()
        preds = model(sent_id,masks)
        loss = criterion(preds,labels)
        train_loss += loss.item()
        
        loss.backward()
        # 梯度剪裁，不太懂
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    print(f'Epoch:{e+1}\t\tTraining Loss: {train_loss / len(train_dataloader)}')

pred_label = []
true_label = []
for batch in tqdm(test_dataloader):
    batch = [i.cuda() for i in batch]
    sent_id,masks,labels = batch 
    
    preds = model(sent_id,masks)
    pred_label.extend(torch.argmax(preds,axis = 1).cpu())
    true_label.extend(labels.cpu())
print(classification_report(true_label, pred_label)) 