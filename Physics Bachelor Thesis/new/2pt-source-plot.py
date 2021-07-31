import matplotlib.pyplot as plt
import numpy as np

#arrays for storing the points
x=[]
y=[]

with open('/home/tzikos/Desktop/new/2pt-source-ad.txt','r',encoding='utf-8') as f:
	lines=f.readlines()	
	for i in lines:
		x.append(float(i.split()[0]))
		y.append(float(i.split()[1])/(297*296/2)*100)


plt.figure()
plt.plot(x,y,'ro')
plt.title("2pt angles in the source data after depropagation")
plt.xlabel("degrees")
plt.ylabel("% of points")
plt.show()

