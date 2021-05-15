from sklearn.datasets import load_boston
import pandas
import csv
from datetime import datetime
import random
def creat_csvfile():
    datestr = (datetime.now()).strftime("%d%b%Y-%H%M%S")
    
    file = 'data-' + datestr +'.csv'
    with open(file,'w',newline="") as f:
        csv_write = csv.writer(f)
        csv_head = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
        csv_write.writerow(csv_head)
    return file

def write_csvfile(fileName,data):
    with open(fileName,'a+',newline="") as f:
        csv_write = csv.writer(f)
        csv_write.writerows(data)
    
fileName=creat_csvfile()
boston = load_boston()
write_csvfile('fileName','boston')