import torch
import requests
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

if response.status_code == 200:
    text = response.text
    # Process the text further as needed
else:
    print("Failed to download the file.")

print(text[:1000])

print("length of dataset in characters: ", len(text))

# This gives us our possible vocabulary for the LLM, see it includes
# capitals and lower case letters, and that we have a total of 65
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# This is an area we could go back for improvement, and try to use chatgpt's "tiktoken" tokenizer or something similar
# create a mapping from characters to integers
# this is where we create our "encoder" and "decoder." I believe in typical LLMs entire
# words are represented, but we are using characters here
stoi = { ch:i for i,ch in enumerate(chars) } # look up table from charcter to integer
itos = { i:ch for i,ch in enumerate(chars) } # now for i:ch
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

# Takes all of the text, encodes it, and wraps it into a torch.tensor to get data
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters look like this to the gpt

# Splitting data into training and validation to 90 to 10 train to test split
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# This is the size of the chunks of texts that are plugged into the transformer
block_size = 8
train_data[:block_size +1 ]

# Generating batches of training data for the transformer with set batch and block size
torch.manual_seed(1337)
batch_size = 4 # of blocks processed in parrallel
block_size = 8 # context length as discussed earlier

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generates batch_size number of random offsets
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

# The above code takes a set batch size and block size, and randomly finds a portion of the text to sample
# So this creates a 4 x 8 matrix, with 4 independent sections of text, and within each independent section of text there are
# 8 dependent sequences of characters
# Tensors are important because it allows you to process data in batches and then move these tensors to be processed on a gpu
        
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C), these are the predictions

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # -log likelihood aka loss function of the predictions and targets

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C) Softmax!
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
# So right now this is a completelly untrained model

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # Many times stotastic gradient decent is used here, but we use AdamW
                                                       # which aparently works well. lr is learning rate

batch_size = 32
for steps in range(10000): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
# At this point, we are only making the prediction from the very last character, we now are going to start
# making predictions based off of more characters

##### Self Attention #####
# This showcases the importance of matrix multiplication

# non-efficient version
# consider the following toy example:
torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels, channel = vocab size
x = torch.randn(B,T,C)
x.shape

# We want x[b,t] = mean_{i<=t} x[b,i]
# bow is bag of words, just averaging all of the previous words
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C) t elements in the path and the two dimension infromation from the tokens
        xbow[b,t] = torch.mean(xprev, 0)



# This is the matrix multiplication trick for weighted aggregation
# This outputs the weighted averages (c) of all of the previous rows of b
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)


# Now this is using matrix multiplication in our original model
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)


# version 3 of above: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf')) # Important, this allows the future not to communicate with the past
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)

# version 4: self-attention! ATTENTION IS ALL YOU NEED PAPER
# This is the good stuff
# Self-attention solves the problem of gathering information from past tokens in a data dependent way
# For instance, if the last vowel was a constinant, than it might weigh past vowels more
# This is done through a "key" and a "query" vector, the key dot products with all the other quieries, signifying
# the relative importance of that token.

torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf')) #to get all the nodes to talk to eachother you only have to delete this line
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x

out.shape