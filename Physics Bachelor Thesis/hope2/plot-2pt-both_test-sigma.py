import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

#THIS IS CODE FOR MAKING THE 2PT PLOTS FOR 2 GRAPHS ON THE SAME PLOT WITH ERRORBARS

#arrays for storing the points
x1=[]
y1=[]
sigma1=[] #sigmas of arrays
x2=[]
y2=[]
sigma2=[] #sigmas of arrays 


with open('/home/tzikos/Desktop/hope2/means/2pt-mean-30-70-bd-new.txt','r',encoding='utf-8') as f1:
	lines1=f1.readlines()	
	for i1 in lines1:
		x1.append(float(i1.split()[0]))
		y1.append(float(i1.split()[1])/(231*230/2)*100)       # the 231*230/2 is the number of my pairs
		sigma1.append(float(i1.split()[2])/(231*230/2)*100)


with open('/home/tzikos/Desktop/hope2/means/2pt-mean-30-70-ad-new.txt','r',encoding='utf-8') as f2:
	lines2=f2.readlines()	
	for i2 in lines2:
		x2.append(float(i2.split()[0]))
		y2.append(float(i2.split()[1])/(231*230/2)*100)
		sigma2.append(float(i2.split()[2])/(231*230/2)*100)



plt.figure()
plt.plot(x1,y1,'r.',x2,y2,'b.')
plt.errorbar(x2,y2,yerr=sigma2,ecolor='b',fmt='none')
plt.errorbar(x1, y1, yerr=sigma1,ecolor='r',fmt='none')
plt.title("2pt angles in the 30-70 case")
plt.xlabel("degrees")
plt.ylabel("% of pairs")
red_patch = mpatches.Patch(color='red', label='Data before depropagation')    #LEGEND-LABEL
blue_patch = mpatches.Patch(color='blue', label='Data after depropagation')   #LEGEND-LABEL
plt.legend(handles=[red_patch,blue_patch])       #this line makes the legend
plt.show()

