import torch, matplotlib.pyplot as plt

with open("names.txt", "r") as file:
  data = file.read().split()

chars = ["/"] + sorted(list(set("".join(data))))
chars_len = len(chars)

## tokenisation
itos = { i: ch for i, ch in enumerate(chars)}
stoi = { ch: i for i, ch in enumerate(chars)}

encode = lambda str : [stoi[x] for x in str]
decode = lambda lst : "".join([itos[x] for x in lst])

## bigram
N = torch.zeros((chars_len, chars_len))
for word in data:
  word = [stoi["/"]] + encode(word) + [stoi["/"]]
  for ix1, ix2 in zip(word, word[1:]):
    N[ix1][ix2] += 1

## norminalised bigram
P = N.float() / torch.sum(N, 1, True)

# sampling
words = []
for i in range(1000):
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, 1, replacement=True).item()
    out.append(itos[ix])
    if ix == 0:
      words.append("".join(out))
      break

# Calculating the loss
loss_likelyhood = 0
n = 0
for word in words:
  for ch1, ch2 in zip(word, word[1:]):
    ix1, ix2 = stoi[ch1], stoi[ch2]
    prob = P[ix1][ix2]
    log_prob = torch.log(prob)
    loss_likelyhood += log_prob
    n += 1

neg_avg_loss_likeleyhood = - loss_likelyhood / n
print(neg_avg_loss_likeleyhood)