import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from collections import OrderedDict


# sentence => 768 embeddings

model = BertModel.from_pretrained('bert-base-uncased',
           output_hidden_states = True,)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# print(tokenizer.encode('[MASK]'))

def bert_text_preparation(text, tokenizer):
    """
    Preprocesses text input in a way that BERT can interpret.
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)
    # convert inputs to tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    return tokenized_text, tokens_tensor, segments_tensor

def get_bert_embeddings(tokens_tensor, segments_tensor, model):
    """
    Obtains BERT embeddings for tokens.
    """
    # gradient calculation id disabled
    with torch.no_grad():
      # obtain hidden states
      outputs = model(tokens_tensor, segments_tensor)
      hidden_states = outputs[2]
    # concatenate the tensors for all layers
    # use "stack" to create new dimension in tensor
    token_embeddings = torch.stack(hidden_states, dim=0)
    # remove dimension 1, the "batches"
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # swap dimensions 0 and 1 so we can loop over tokens
    token_embeddings = token_embeddings.permute(1,0,2)
    # intialized list to store embeddings
    token_vecs_sum = []
    # "token_embeddings" is a [Y x 12 x 768] tensor
    # where Y is the number of tokens in the sentence
    # loop over tokens in sentence
    for token in token_embeddings:
    # "token" is a [12 x 768] tensor
    # sum the vectors from the last four layers
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum

sentences = []
with open('human.txt', 'r') as file:
    lines = file.readlines()
sentences = lines
with open('concrete.txt', 'r') as file:
    lines = file.readlines()
sentences += lines


context_embeddings = []
context_tokens = []
target_token = 103
for sentence in sentences[:10]:
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
    list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    #   make ordered dictionary to keep track of the position of each   word
    #   loop over tokens in sensitive sentence
    # print(tokenized_text)
    for index, token in enumerate(tokenized_text):
        if token == "[MASK]":
            target_index = index
            break
    context_embeddings.append(list_token_embeddings[target_index].detach().numpy())

for embed in context_embeddings:
    print(embed.shape)

array = np.asarray(context_embeddings)
print(array.shape)
  




  
#   for token in tokenized_text[1:-1]:
#     # keep track of position of word and whether it occurs multiple times
#     if token in tokens:
#       tokens[token] += 1
#     else:
#       tokens[token] = 1
#   # compute the position of the current token
#     token_indices = [i for i, t in enumerate(tokenized_text) if t == token]
#     current_index = token_indices[tokens[token]-1]
#   # get the corresponding embedding
#     token_vec = list_token_embeddings[current_index]
    
#     # save values
#     context_tokens.append(token)
#     context_embeddings.append(token_vec)
