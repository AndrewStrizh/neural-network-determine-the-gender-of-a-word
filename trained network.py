from main import *
import torch
import random
words, classes = load_dictionary('russian_nouns.txt')
s = input('want to retrain model? yes/no ')
nn = torch.load('bumba.txt')
if s == 'yes':
  epochs = int(input("epochs amount "))
  speed = float(input("training speed "))
  nn = train_more(nn, epochs, words, classes, speed)
while True:
  word = input()
  ind = nn(word_onehot(word)).tolist().index(max(nn(word_onehot(word)).tolist()))
  if ind == 0: print("Мужской род")
  if ind == 1: print("Женский род")
  if ind == 2: print("Средний род")
