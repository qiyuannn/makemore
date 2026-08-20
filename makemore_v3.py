import torch, matplotlib.pyplot as plt

with open("names.txt", "r") as file:
  data = file.read().split()

chars = ["/"] + sorted(list(set("".join(data))))
len_chars = len(chars)

# tokeniser
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

encode = lambda str: [stoi[x] for x in str]
decode = lambda lst: "".join([itos[x] for x in lst])

# building the dataset
block_size = 3

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
dimensions = 10

C = torch.randn((len_chars, dimensions), requires_grad=True)
W1 = torch.randn((dimensions * block_size, 200), requires_grad=True)
h1 = torch.randn(200, requires_grad=True)
W2 = torch.randn((200, len_chars), requires_grad=True)
h2 = torch.randn(len_chars, requires_grad=True)
parameters = [C, W1, h1, W2, h2]

for i in range(200000):
  #foward pass
  ix = torch.randint(0, X_train.shape[0], (32,))
  emb = C[X_train[ix]]
  h = torch.tanh(emb.view(-1, dimensions * block_size) @ W1 + h1)
  logits = h @ W2 + h2

  #loss
  loss = torch.nn.functional.cross_entropy(logits, Y_train[ix])

  #backward pass
  for p in parameters:
    p.grad = None

  loss.backward()


  lr = 0.01 if i < 100000 else 0.01
  for p in parameters:
    p.data -= lr * p.grad 

  if i % 10000 == 0:
    print(f"interation: {i} loss: {loss}")


with torch.no_grad():
  emb = C[X_val]
  h = torch.tanh(emb.view(-1, block_size * dimensions) @ W1 + h1)
  logits = h @ W2 + h2
  loss = torch.nn.functional.cross_entropy(logits, Y_val)
  print(f"val loss: {loss}")
