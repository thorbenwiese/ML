# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

'''
05.1.a)
'''
def loadMatFile():
  data = sio.loadmat('iris_multiclass.mat')
  print data['meas']
  print data['indices_test']
  print data['indices_train']
  print data['species']









def main():
  loadMatFile()










# execute main
if __name__ == "__main__":
  main()
