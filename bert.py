import re
import torch
from transformers import BertTokenizer, BertModel,BertConfig,AdamW
from torch.utils.data import TensorDataset,DataLoader,Dataset
import torch.nn as nn
from tqdm import tqdm
import wandb
from sklearn import metrics
import numpy as np
""""
由于显卡内存不够(3050ti),无法进行训练
因为没有跑过,不知道有没有BUG
"""
class Dataset(Dataset):
    def __init__(self,filename):
        self.labels=['0','1']
        self.labels_id=[0,1]
        self.input_ids=[]
        self.token_type_ids=[]
        self.attention_mask=[]
        self.label_id=[]
        self.load_data(filename)

    def load_data(self,filename):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        with open(filename, 'r',encoding='utf-8',errors="ignore") as f:
            data = f.readlines()
        
        for item in data:
            item=re.sub('\s|\t|\n',"",item)
            label=item[0]
            text=item[1:]
            label_id=self.labels.index(label)
            token = tokenizer(text,add_special_tokens=True, max_length=512, padding='max_length', truncation=True)
            self.input_ids.append(np.array(token['input_ids']))
            self.attention_mask.append(np.array(token['attention_mask']))
            self.token_type_ids.append(np.array(token['token_type_ids']))
            self.label_id.append(label_id)

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_id[index]

    def __len__(self):
        return len(self.input_ids)
        
class Bertclassifier(nn.Module):
    def __init__(self,bert_config,num_labels):
        super().__init__()
        self.bert=BertModel(config=bert_config)
        self.classifier=nn.Linear(bert_config.hidden_size,num_labels)
        self.softmax=nn.functional.softmax
        self.__name__='BERT'
    def forward(self,input_ids,attention_mask,token_type_ids):
        bert_output=self.bert(input_ids=input_ids,attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        pooled=bert_output[1]
        logits=self.classifier(pooled)
        return self.softmax(logits,1)
    
def run():
    DEVICE=torch.device('cpu')

    train_data=Dataset('Dataset/train.txt')
    valid_data=Dataset('Dataset/validation.txt')
    test_data=Dataset('Dataset/test.txt')
    batch_size=50
    epochs=10
    learning_rate=1e-3

    train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
    valid_dataloader=DataLoader(valid_data,batch_size=batch_size,shuffle=True)
    test_dataloader=DataLoader(test_data,batch_size=batch_size,shuffle=True)
    bert_config=BertConfig.from_pretrained('bert-base-chinese')
    num_labels=2

    model=Bertclassifier(bert_config,num_labels).to(DEVICE)

    optimizer=AdamW(model.parameters(),lr=learning_rate)

    criterion=nn.CrossEntropyLoss()

    best_f1=0

    wandb.init(project=f"hw2",name=f"{model.__name__}",entity="zengxw21")
    wandb.config={"learning_rate":0.001,"epochs":100,"batch_size":50}
    for epoch in tqdm(range(1,epochs+1)):
        losses=0
        accuracy=0

        model.train()
        
        train_bar=tqdm(train_dataloader,ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            output = model(
                input_ids=input_ids.to(DEVICE), 
                attention_mask=attention_mask.to(DEVICE), 
                token_type_ids=token_type_ids.to(DEVICE), 
            )

            loss=criterion(output,label_id.to(DEVICE))
            losses+=loss.item()

            pred_labels=torch.argmax(output,dim=1)
            acc=torch.sum(pred_labels==label_id.to(DEVICE)).item()/len(pred_labels)

            accuracy+=acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(),acc=acc)
            
        train_loss = losses / len(train_dataloader)
        train_acc = accuracy / len(train_dataloader)

        print('\tTrain ACC:', train_acc, '\tLoss:', train_loss)
           
        model.eval()
        losses=0
        accuracy=0
        pred_labels = []
        true_labels = []
        valid_bar = tqdm(valid_dataloader, ncols=100)
        with torch.no_grad():
            for input_ids, token_type_ids, attention_mask, label_id  in valid_bar:
                valid_bar.set_description('Epoch %i valid' % epoch)

                output = model(
                    input_ids=input_ids.to(DEVICE), 
                    attention_mask=attention_mask.to(DEVICE), 
                    token_type_ids=token_type_ids.to(DEVICE), 
                )

                loss = criterion(output, label_id.to(DEVICE))
                losses += loss.item()

                pred_label = torch.argmax(output, dim=1)   # 预测出的label
                acc = torch.sum(pred_label == label_id.to(DEVICE)).item() / len(pred_label) #acc
                accuracy+=acc
                
                valid_bar.set_postfix(loss=loss.item(), acc=acc)

                pred_labels.extend(pred_label.cpu().numpy().tolist())
                true_labels.extend(label_id.numpy().tolist())
        
        valid_loss = losses / len(valid_dataloader)
        valid_acc = accuracy / len(valid_dataloader)
        print('\tVal ACC:', valid_acc,'\tLoss:', valid_loss)

        model.eval()
        losses=0
        accuracy=0
        pred_labels = []
        true_labels = []
        test_bar = tqdm(test_dataloader, ncols=100)
        with torch.no_grad():
            for input_ids, token_type_ids, attention_mask, label_id  in test_bar:
                test_bar.set_description('Epoch %i valid' % epoch)

                output = model(
                    input_ids=input_ids.to(DEVICE), 
                    attention_mask=attention_mask.to(DEVICE), 
                    token_type_ids=token_type_ids.to(DEVICE), 
                )

                loss = criterion(output, label_id.to(DEVICE))
                losses += loss.item()

                pred_label = torch.argmax(output, dim=1)   # 预测出的label
                acc = torch.sum(pred_label == label_id.to(DEVICE)).item() / len(pred_label) #acc
                accuracy+=acc
                
                test_bar.set_postfix(loss=loss.item(), acc=acc)

                pred_labels.extend(pred_label.cpu().numpy().tolist())
                true_labels.extend(label_id.numpy().tolist())
        
        test_loss = losses / len(test_dataloader)
        test_acc = accuracy / len(test_dataloader)
        print('\ttest ACC:', test_acc,'\tLoss:', test_loss)

        wandb.log(
            {
                
                #"train_f1": train_f1,
                "val_loss": valid_loss,
                "val_acc": valid_acc,
                #"val_f1": val_f1,
                "test_loss": test_loss,
                "test_acc": test_acc,
                #"test_f1": test_f1,
            }
        )