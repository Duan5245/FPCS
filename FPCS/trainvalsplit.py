import pandas as pd
import random
import pdb
if __name__ == "__main__":
    scan05 = pd.read_csv('all.txt', header=None, sep=' ')
    lenscan = len(scan05)
    allidxs = [i for i in range(lenscan)]
    random.shuffle(allidxs)
    trainidxs = allidxs[:int(lenscan*0.9)]
    testidxs = allidxs[int(lenscan*0.9):]

    train = scan05.loc[trainidxs,:]
    test = scan05.loc[testidxs,:]
    train.to_csv('train.txt', sep=' ', header=0, index=0)
    test.to_csv('test.txt', sep=' ', header=0,index=0)
