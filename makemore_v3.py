import torch

with open("names.txt", "r") as file:
  data = file.read().split()

chars = ["/"] + sorted(list(set("".join(data))))
num_chars = len(chars)

# tokenisation
itos = { i: ch for i, ch in enumerate(chars)}
stoi = { ch: i for i, ch in enumerate(chars)}

encode = lambda str: [stoi[x] for x in str]
decode = lambda lst: "".join([itos[x] for x in lst])

# dataset
block_size = 3

def build_dataset(data):
  X, Y = [], []
  for word in data:
    context = [0] * block_size
    for char in word + "/":
      ix = stoi[char]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix]
  X, Y = torch.tensor(X), torch.tensor(Y)
  return X, Y

c1, c2 = int(len(data) * 0.8), int(len(data) * 0.9)
X_train, Y_train = build_dataset(data[:c1])
X_val, Y_val = build_dataset(data[c1: c2])
X_test, Y_test = build_dataset(data[c2:])

C = torch.randn((num_chars, 2), requires_grad=True)
W1 = torch.randn((block_size * 2, 300), requires_grad=True)
b1 = torch.randn(300, requires_grad=True) 
W2 = torch.randn((300, num_chars), requires_grad=True)
b2 = torch.randn(num_chars, requires_grad=True)
parameters = [C, W1, b1, W2, b2]

for _ in range(10000):
  ix = torch.randint(0, X_train.shape[0], (32,))

  emb = C[X_train[ix]]
  h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
  logits = h @ W2 + b2
  loss = torch.nn.functional.cross_entropy(logits, Y_train[ix])

  for p in parameters:
    p.grad = None

  loss.backward()

  with torch.no_grad():
    for p in parameters:
      p -= 0.01 * p.grad

print(f"train loss: {loss}")

with torch.no_grad():
  emb = C[X_val]
  h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
  logits = h @ W2 + b2
  loss = torch.nn.functional.cross_entropy(logits, Y_val)

  print(f"val loss: {loss}")