from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe, CharNGram


# Approach 1:
# set up fields
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)


# make splits for data
train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
LABEL.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())
a=TEXT.vocab
# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=3)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.label)