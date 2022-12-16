
import string
from Converter import BertTokenLabelConverter
from Dataset import TextData
import argparse
import numpy as np
from transformers import PreTrainedModel,AutoTokenizer,default_data_collator,RobertaTokenizer,RobertaForMaskedLM,RobertaModel
import torch
from torch.utils.data import Dataset,DataLoader

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    train_loss = 0
    for idx,(ids,attention_mask) in enumerate(train_loader):
        ids,attention_mask = ids.to(device),attention_mask.to(device)
        optimizer.zero_grad()
        out = model(input_ids = ids,attention_mask = attention_mask)
        out_logits = out.logits
        criterion = torch.nn.CrossEntropyLoss(weight=1-attention_mask,ignore_index=0)
        loss = criterion(out_logits,ids)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*ids.shape[0]
        if idx % 100 == 0:
            print(f'At epoch {epoch+1}, step {idx+1}, train loss is {train_loss/(ids.shape[0]*(idx+1))}')

    train_loss /= len(train_loader.dataset)

    return train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (ids, attention_mask) in enumerate(test_loader):
            ids, attention_mask = ids.to(device), attention_mask.to(device)
            out = model(input_ids=ids, attention_mask=attention_mask)
            out_logits = out.logits
            criterion = torch.nn.CrossEntropyLoss(weight=1 - attention_mask, ignore_index=0)
            loss = criterion(out_logits, ids)
            test_loss += loss.item() * ids.shape[0]

    test_loss /= len(test_loader.dataset)

    return test_loss


if __name__ == '__main__':
    converter = BertTokenLabelConverter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('texts.txt') as f:
        tgt = f.readlines()

    text = []
    for i in tgt:
        text.append(i.strip('\n'))

    text_len = len(text)
    train_set = TextData(text[:int(0.8*text_len)],converter = BertTokenLabelConverter())
    test_set = TextData(text[int(0.8*text_len):],converter = BertTokenLabelConverter(is_train=False))
    num_epoch = 200
    model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)
    model.roberta.embeddings.word_embeddings = torch.nn.Embedding(96, 768, padding_idx=1)
    model.lm_head.decoder = torch.nn.Linear(768, 96)

    try:
        model = torch.load('.\Roberta.pt')
    except:
        print('saved model not found')
    train_loader = DataLoader(dataset=train_set,batch_size=128,shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False)
    optimizer = torch.Adam(model.parameters(),lr = 1e-3)
    test_losses = []
    for epoch in range(num_epoch):
        train_loss = train(model,device,train_loader,optimizer,epoch)
        test_loss = test(model,device,test_loader)
        if test_loss < min(test_losses):
            torch.save(model, '.\Roberta.pt')
        test_losses.append(test_loss)
        print(f'At epoch {epoch+1},train loss is {train_loss}')
        print(f'At epoch {epoch+1},test loss is {test_loss}')


