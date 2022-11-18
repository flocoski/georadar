import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import struct

path_rad=""

#Décomposition d'un RAD
rad=open(path_rad,"r")
rad1=rad.readlines()
rad.close()
param={}
for line in rad1:
    line=line.strip('\n')
    line=line.split(':')
    param[str(line[0])]=line[1]

nbTraces=int(param["LAST TRACE"]) #nombre de traces
nbSamples=int(param["SAMPLES"]) #nombre de sample
timewindow=float(param["TIMEWINDOW"]) #temps de mesure pour 1 trace
dx=float(param["DISTANCE INTERVAL"]) #distance entre 2 traces
dt=timewindow/nbSamples
lt=[k*dt for k in range(nbSamples)]

with open(path_rad('.rad')+".rd3", mode='rb') as rd3data:
    rd3data=rd3data.read()
rd3=struct.unpack("h"*((len(rd3data))//2), rd3data)
rd3=np.reshape(rd3,(nbTraces(),nbSamples())) 
rd3=np.transpose(rd3)

#Affichage d'un RD3
img=plt.imshow(rd3, interpolation='nearest', aspect = 'auto', cmap='seismic', extent=[0,nbTraces,lt[-1],0])
plt.xlabel("traces")
plt.ylabel("temps (ns)") 
plt.show()
