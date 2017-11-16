import matplotlib.pyplot as plt
import numpy as np
 
with open("./expe/nis.csv") as f:
  line = f.readline()
  radar_l = []
  laser_l = []
  while line:
    if line[:3]!="NIS":
      line = f.readline()
      continue
    _, type, nis = line.split(",")
    if type=="radar":
      radar_l.append(float(nis))
    if type=="laser":
      laser_l.append(float(nis))
    line = f.readline()
   
plt.hlines(y=7.8, xmin=0, xmax=260)
plt.plot(radar_l)
plt.title("radar: "+str(np.mean(np.array(radar_l)>7.8)))
plt.show()

plt.hlines(y=7.8, xmin=0, xmax=260)
plt.plot(laser_l) 
plt.title("laser: "+str(np.mean(np.array(radar_l)>7.8)))
plt.show()
