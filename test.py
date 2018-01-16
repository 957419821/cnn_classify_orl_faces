import numpy as np
from dataset import *
from dataSetTools import tools
import csv

batchStart = 0
batchEnd = 10
datas = tools.getData('orl_faces/', batchStart, batchEnd)
out = open("datas1.csv", 'w', newline='')
writer = csv.writer(out, dialect='excel')
for r in datas:
    writer.writerow(r)
