import math
import torch
import torch.nn.functional as F

with open("names.txt", "r") as file:
  data = file.read().split()

data = [data[i] for i in torch.randperm(len(data)).tolist()]

chars = ["/"] + sorted(list(set("".join(data))))
len_chars = len(chars)

# tokeniser
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

encode = lambda str: [stoi[x] for x in str]
decode = lambda lst: "".join([itos[x] for x in lst])

# building the dataset
block_size = 5

def build_data(data):
  x, y = [], []
  for words in data:
    context = [0] * block_size
    for char in words + "/":
      ix = stoi[char]
      x.append(context)
      y.append(ix)
      context = context[1:] + [ix]
  return torch.tensor(x), torch.tensor(y)

n1 = int(len(data) * 0.8)
n2 = int(len(data) * 0.9)
X_train, Y_train = build_data(data[:n1])
X_val, Y_val = build_data(data[n1: n2])
X_test, Y_test = build_data(data[n2:])

# neural net parameters
dimensions = 16
hidden_units = 300
max_steps = 100000

C = torch.randn((len_chars, dimensions))
W1 = torch.randn((dimensions * block_size, hidden_units)) * (5/3) / math.sqrt(dimensions * block_size)
h1 = torch.randn(hidden_units) * 0.01
W2 = torch.randn((hidden_units, len_chars)) * 0.01
h2 = torch.zeros(len_chars)
parameters = [C, W1, h1, W2, h2]
for p in parameters:
  p.requires_grad = True

for i in range(max_steps):
  #foward pass
  ix = torch.randint(0, X_train.shape[0], (128,))
  emb = C[X_train[ix]]
  h = torch.tanh(emb.view(-1, dimensions * block_size) @ W1 + h1)
  logits = h @ W2 + h2

  #loss
  loss = F.cross_entropy(logits, Y_train[ix])

  #backward pass
  for p in parameters:
    p.grad = None

  loss.backward()

  lr = 0.1 if i < int(max_steps * 0.75) else 0.01
  for p in parameters:
    p.data -= lr * p.grad 

  if i % 10000 == 0:
    print(f"iteration: {i} loss: {loss.item():.4f}")

@torch.no_grad()
def split_loss(split, X, Y):
  emb = C[X]
  h = torch.tanh(emb.view(-1, block_size * dimensions) @ W1 + h1)
  logits = h @ W2 + h2
  loss = F.cross_entropy(logits, Y)
  print(f"{split} loss: {loss.item():.4f}")

split_loss("train", X_train, Y_train)
split_loss("val", X_val, Y_val)
split_loss("test", X_test, Y_test)
