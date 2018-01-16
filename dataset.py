from dataSetTools.tools import *
import numpy as np
import random
import csv
np.lookfor('matlib')

def getCSVdata(csvfile):
    print("--------------------")
    print("getting datas from CSV file")
    print("--------------------")
    reader = csv.reader(open(csvfile, encoding='utf-8'))
    data = []
    for r in reader:
        for i in range(len(r)-1): # 最后一行是空格
            r[i] = int(float(r[i]))
        label = []
        for i in range(1, len(r[-1])-1):
            if r[-1][i] == '0':
                label.append(0)
            if r[-1][i] == '1':
                label.append(1)
        r[-1] = label
        data.append(r)
    return data

class DataSet():
    def __init__(self):
        self.height = 112
        self.width = 92
        self.csvfile = 'datas1.csv'
        self.datas = getCSVdata(self.csvfile)
    def nextBatch(self):
        x = []; y = []
        random.shuffle(self.datas)
        for i in range(500):
            x.append(self.datas[i][:-1])
            y.append(self.datas[i][-1])
        batch = [x, y]
        return batch

        
