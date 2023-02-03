from main import *
import torch
import random
nn = torch.load('bumba.txt')
while True:
  word = input()
  ind = nn(word_onehot(word)).tolist().index(max(nn(word_onehot(word)).tolist()))
  if ind == 0: print("Мужской род")
  if ind == 1: print("Женский род")
  if ind == 2: print("Средний род")
