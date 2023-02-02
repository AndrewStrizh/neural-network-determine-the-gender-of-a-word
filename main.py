import torch
import random
class DenseLayer:
    def __init__(self, activation, input_size, output_size):
        self.weights = torch.normal(mean = 0.0, std = 2.0/(input_size + output_size)**0.5, size=(input_size, output_size))
        self.weights.requires_grad_()

        self.bias = torch.zeros(output_size)
        self.bias.requires_grad_()

        self.activation = activation

    def __call__(self, inputs):
        return self.activation(torch.tensordot(inputs, self.weights, dims=1) + self.bias)
  
    def update_weights(self, learning_rate):
        with torch.no_grad(): ## отключает построение графа вычислений
            self.weights -= self.weights.grad * learning_rate
            self.weights.grad = None
            
            self.bias -= self.bias.grad * learning_rate
            self.bias.grad = None

def relu(x):
    return (x > 0) * x

def softmax(x):
    return x.exp() / x.exp().sum(-1)

def cross_entropy(x, y, BATCH_SIZE = 50):
    N = BATCH_SIZE
    loss = -1 * (1 / N) * (torch.tensordot(y, x.log()))
    return loss


class Network:
    def __init__(self, input_size, layer_sizes, output_size):
        self.layers = []
        prev_size = input_size
        for size in layer_sizes:
            self.layers.append(DenseLayer(relu, prev_size, size))
            prev_size = size

        self.layers.append(DenseLayer(softmax, prev_size, output_size))
    
    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def train(self, words, classes, learning_rate):
        train_words = self(words[0]).clone()
        train_words = torch.reshape(train_words, (1, 3,))
        for i in range(1, len(words)):
            word = self(words[i])
            word = torch.reshape(word, (1, 3,))
            train_words = torch.cat((train_words, word))
        loss = cross_entropy(train_words, classes)
        loss.backward()
        for layer in self.layers:
            layer.update_weights(learning_rate)
        return loss.item()

def load_dictionary(filename):
    words, classes = [], []
    with open(filename, encoding='utf8') as f:
        for line in f.readlines():
            line = line.split()
            words.append(word_onehot(line[0]))
            classes.append(cls_onehot(line[1]))
    return words, classes

def clean_word(word):
    punctuation = '.,;:"!?""_-'
    word = word.lower()
    word = word.strip(punctuation)
    return word

def word_onehot(word, ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя', MAX_LEN = 20):
    ALPHABET_SIZE = len(ALPHABET)
    word = clean_word(word)
    word_encoded = torch.zeros(MAX_LEN * ALPHABET_SIZE)
    for i in range(len(word)):
        if i > MAX_LEN-1: break
        word_encoded[i * ALPHABET_SIZE + ALPHABET.index(word[len(word) - 1 - i])] = 1.
    return word_encoded

def cls_onehot(cls, size = 3):
    cls = int(cls)
    cls_encoded = [0.] * size
    cls_encoded[cls] = 1.
    return cls_encoded

def gen_batch(words,classes, BATCH_SIZE = 50):
    index = random.randint(0, len(classes) - BATCH_SIZE-1)
    words = words[index:index + BATCH_SIZE]
    classes = classes[index:index + BATCH_SIZE]
    return words, classes

def main():
    words, classes = load_dictionary('russian_nouns.txt')
    classes = torch.tensor(classes)
    MAX_LEN= 20
    nn = Network(33 * MAX_LEN, [20, 15, 10, 15, 20], 3)
    for epoch in range(5001):
        words_batch, classes_batch = gen_batch(words, classes)
        loss = nn.train(words_batch, classes_batch, 0.01)
        if epoch % 500 == 0:
            print(f'completed {epoch//50}%', 'loss = ', loss)
    print('Training completed')
    torch.save(nn, 'bumba.txt')
    while True:
        word = input()
        ind = nn(word_onehot(word)).tolist().index(max(nn(word_onehot(word)).tolist()))
        if ind == 0: print("Мужской род")
        if ind == 1: print("Женский род")
        if ind == 2: print("Средний род")

if __name__ == "__main__":
    main()