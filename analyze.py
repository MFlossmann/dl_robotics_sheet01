#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import pandas as pd

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage python analyze.py <data.csv>")
        exit(1)

    df = pd.read_csv(sys.argv[1],
                     sep='\t')

    print(df.columns)

    train_loss = np.array(df.get('train_loss').dropna())
    train_epoch = np.array(df.get('train_epoch').dropna())
    eval_loss = np.array(df.get('eval_loss').dropna())
    eval_epoch = np.array(df.get('eval_epoch').dropna())

    plt.figure('Learning Curve')
    plt.plot(train_epoch,
             train_loss,
             'r-',
             label="Train losses")
    plt.plot(eval_epoch,
             eval_loss,
             'b-',
             label='Eval losses')
    plt.legend()

    try:
        plt.show()
    except:
        plt.savefig(sys.argv[1] + ".png")
