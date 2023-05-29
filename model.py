import torch 
import torch.nn as nn
import torch.nn.functional as F
from read_data import Word2Id,Word2Vec
import math

word2id=Word2Id()
word2vec=Word2Vec('wiki',word2id)

class HyperParam:
    # 训练中更新Word2Vec
    update=True
    vocab_size=len(word2id)+1
    classify_num=2
    embedding_dim=50
    dropout_keep=0.3
    kernel_num=20
    kernel_size=[3,5,7]
    pretrained_embedding=word2vec
    hidden_size=100
    hidden_layers=2

class textCNN(nn.Module):
    def __init__(self,config:HyperParam): 
        super(textCNN,self).__init__()
        update=config.update
        vocab_size=config.vocab_size
        n_class=config.classify_num
        embedding_dim=config.embedding_dim
        kernel_num=config.kernel_num
        kernel_size=config.kernel_size
        drop_keep_prob=config.dropout_keep
        pretrained_embed=config.pretrained_embedding

        self.__name__='textCNN'

        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight.requires_grad=update
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.conv1=nn.Conv2d(1,kernel_num,(kernel_size[0],embedding_dim))
        self.conv2=nn.Conv2d(1,kernel_num,(kernel_size[1],embedding_dim))
        self.conv3=nn.Conv2d(1,kernel_num,(kernel_size[2],embedding_dim))

        self.dropout=nn.Dropout(drop_keep_prob)

        self.fc=nn.Linear(len(kernel_size)*kernel_num,n_class)

    @staticmethod
    def conv_and_pool(x,conv):
        x=F.relu(conv(x).squeeze(3))
        return F.max_pool1d(x,x.size(2)).squeeze(2)
    
    def forward(self,x):
        x=self.embedding(x.to(torch.int64)).unsqueeze(1)
        x1=self.conv_and_pool(x,self.conv1)
        x2=self.conv_and_pool(x,self.conv2)
        x3=self.conv_and_pool(x,self.conv3)
        out=torch.cat((x1,x2,x3),1)
        out=self.dropout(out)
        out=self.fc(out)
        out=F.log_softmax(out,dim=1)
        return out
    
class RNN_GRU(nn.Module):
    def __init__(self,config:HyperParam):
        super(RNN_GRU,self).__init__()

        update=config.update
        vocab_size=config.vocab_size
        n_class=config.classify_num
        embedding_dim=config.embedding_dim
        pretrained_embed=config.pretrained_embedding
        self.num_layers=config.hidden_layers
        self.hidden_size=config.hidden_size
        self.__name__='RNN_GRU'

        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight.requires_grad=update
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))


        self.encoder=nn.GRU(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=config.dropout_keep# add dropout
        )
        self.decoder=nn.Linear(2*self.hidden_size,64)
        self.fc=nn.Linear(64,n_class)

    def forward(self,inputs):
        x=self.embedding(inputs.to(torch.int64)).permute(1,0,2)
        h_0 = torch.rand(self.num_layers * 2, x.size(1), self.hidden_size).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        _, h_n = self.encoder(x, h_0)          # (num_layers * 2, batch, hidden_size)
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)    # view h_n as (num_layers, num_directions, batch, hidden_size)
        x=torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)
        x=self.decoder(x)
        x=self.fc(x)
        return x

class RNN_LSTM(nn.Module):
    def __init__(self,config:HyperParam):
        super(RNN_LSTM,self).__init__()

        vocab_size = config.vocab_size
        update_w2v = config.update
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embedding
        self.num_layers = config.hidden_layers
        self.hidden_size = config.hidden_size
        self.n_class = config.classify_num
        self.__name__ = 'RNN_LSTM'

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.encoder=nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=config.dropout_keep#add dropout
        )

        self.decoder=nn.Linear(2*self.hidden_size,64)
        self.fc1=nn.Linear(64,self.n_class)

    def forward(self,inputs):
        _, (h_n, _) = self.encoder(self.embedding(inputs.to(torch.int64)).permute(1, 0, 2))  # (num_layers * 2, batch, hidden_size)
        # view h_n as (num_layers, num_directions, batch, hidden_size)
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
        x=torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)
        x=self.decoder(x)
        x=self.fc1(x)
        return x

class LSTM_Attention(nn.Module):
    def __init__(self,config:HyperParam):
        super(LSTM_Attention,self).__init__()
        vocab_size=config.vocab_size
        update_w2v=config.update
        embedding_dim=config.embedding_dim
        pretrained_embed=config.pretrained_embedding
        self.num_layers=config.hidden_layers
        self.hidden_size=config.hidden_size
        self.n_class=config.classify_num
        self.dropout=config.dropout_keep
        self.__name__="LSTM_Attention"

        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight.requires_grad=update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.encoder=nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.weight_W=nn.Parameter(torch.Tensor(2*self.hidden_size,2*self.hidden_size))
        self.weight_proj=nn.Parameter(torch.Tensor(2*self.hidden_size,1))

        nn.init.uniform_(self.weight_W,-0.1,0.1)
        nn.init.uniform_(self.weight_proj,-0.1,0.1)

        self.decoder1=nn.Linear(self.hidden_size*2,self.hidden_size)
        self.decoder2=nn.Linear(self.hidden_size,self.n_class)

    def forward(self,inputs):
        embeddings=self.embedding(inputs.to(torch.int64))
        states,hidden=self.encoder(embeddings.permute([0,1,2]))

        u=torch.tanh(torch.matmul(states,self.weight_W))
        att=torch.matmul(u,self.weight_proj)

        att_score=F.softmax(att,dim=1)

        scored_x=states*att_score
        encoding=torch.sum(scored_x,dim=1)

        outputs=self.decoder1(encoding)
        outputs=self.decoder2(outputs)
        return outputs

class MLP(nn.Module):
    def __init__(self,config:HyperParam) :
        super(MLP,self).__init__()
        vocab_size=config.vocab_size
        update_w2v=config.update
        embedding_dim=config.embedding_dim
        pretrained_embed=config.pretrained_embedding
        self.num_layers=config.hidden_layers
        hidden_size=config.hidden_size
        n_class=config.classify_num
        self.__name__='MLP'
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight.requires_grad=update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.relu=torch.nn.ReLU()
        self.mlp_layer=torch.nn.Linear(embedding_dim,hidden_size)
        self.linear=torch.nn.Linear(hidden_size,n_class)
        self.dropout=nn.Dropout(p=0.5)
        for _,p in self.named_parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p,mean=0,std=0.01)

    def forward(self,inputs):
        output=self.relu(self.mlp_layer(self.embedding(inputs.to(torch.int64)))).permute(0,2,1)
        x= self.linear(F.max_pool1d(output,output.shape[2]).squeeze(2))
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self,d_model, vocab_size, dropout) :
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)

        pe=torch.zeros(vocab_size,d_model)
        position=torch.arange(0,vocab_size,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(
            torch.arange(0,d_model,2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x=x+self.pe[:,:x.size(1),:]
        return self.dropout(x)
    
class transformer(nn.Module):
    def __init__(
            self,
            config:HyperParam,
            nhead=5,
            activation="relu",
            classifier_dropout=0.1):
        super(transformer,self).__init__()

        vocab_size=config.vocab_size
        d_model=config.embedding_dim
        self.__name__="transformer"
        assert d_model%nhead==0, "nheads must divide evenly into d_model"

        self.emb=nn.Embedding(vocab_size,d_model,padding_idx=0)

        self.pos_encoder=PositionalEncoding(
            d_model=d_model,
            dropout=config.dropout_keep,
            vocab_size=vocab_size
        )

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=config.dropout_keep
        )
        self.transforemer_encoder=nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.hidden_layers
        )
        self.classifier=nn.Linear(d_model,2)
        self.d_model=d_model

    def forward(self,x):
        x=x.to(torch.int64)
        x=self.emb(x)*math.sqrt(self.d_model)
        x=self.pos_encoder(x)
        x=self.transforemer_encoder(x)
        x=x.mean(dim=1)
        x=self.classifier(x)
        return x
