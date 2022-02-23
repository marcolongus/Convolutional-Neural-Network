import numpy as np
import matplotlib.pyplot as plt


accuracy = [89.0, 89.0, 87.0, 81.0, 91.0, 90.0, 82.0, 79.0]
acc_txt =  [89.4, 88.5, 86.5, 81.3, 90.3, 89.5, 82.3, 79.2]

names = [
'M(Red 1, ADAM, CEL)', 
'M(Red 1, ADAM, MSE)' , 
'M(Red 1, SGD, CEL)',
'M(Red 1, SGD, MSE)' ,
'M(Red 2, ADAM, CEL)' ,
'M(Red 2, ADAM, MSE)' ,
'M(Red 2, SGD, CEL)' ,
'M(Red 2, SGD, MSE)' ,
 ]

points = zip([i for i in range(8)], accuracy)
point_txt = zip([i for i in range(1,9)], acc_txt)

color = ['blue', 'red', 'green', 'orange', 'black', 'c', 'purple', 'gray']

for i in range(1,9):
	plt.scatter(i, accuracy[i-1], marker=".", color = color[i-1])

plt.xlim(0,14)
for i, (name, point, p_txt) in enumerate(zip(names, points, point_txt)):
	print(name, point, p_txt)
	plt.annotate(name, xy= point , xytext=p_txt, color=color[i], size='x-large')
plt.show()