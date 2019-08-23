import gzip
import numpy as np

f = gzip.open('MNIST_data/train-labels-idx1-ubyte.gz','r')
f.read(8)
for i in range(0,50):
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(labels)