import torch

# By gradient based optimisation

with open("names.txt", "r") as file:
  data = file.read().split()

chars = ["/"] + sorted(list(set("".join(data))))
len_chars = len(chars)

# tokenisation
stoi = { ch: i for i, ch in enumerate(chars)}
itos = { i: ch for i, ch in enumerate(chars)}
encode = lambda str : [stoi[x] for x in str]
decode = lambda lst: ''.join([itos[x] for x in lst])

# creating the training set
xs, ys = [], []
for word in data:
  word = encode('/' + word + '/')
  for ix1, ix2 in zip(word, word[1:]):
    xs.append(ix1)
    ys.append(ix2)
xs, ys = torch.tensor(xs), torch.tensor(ys)

# creating a neural network
W = torch.randn((27, 27), requires_grad=True)

x_encoded = torch.nn.functional.one_hot(xs, num_classes=27).float()

for k in range(1000):
  # foward pass 
  logits = x_encoded @ W # log counts
  counts = logits.exp() # softmax
  probs = counts / torch.sum(counts, 1, keepdim=True)
  neg_loss_likelyhood = -probs[torch.arange(len(ys)), ys].log().mean() + 0.01 * (W ** 2).mean() # with smoothing 

  # backward pass
  W.grad = None
  neg_loss_likelyhood.backward()

  W.data += -10 * W.grad

  if k == 1000 - 1:
    print(neg_loss_likelyhood)

  
