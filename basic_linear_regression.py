import numpy as np
x=np.array([1,2,3,4,5,])
y=np.array([5,7,9,11,13])
mi=bi=0
learning_rate =0.1
n=len(x)
costs=0
ite=1000
for i in range(ite):
    yp = mi * x + bi
    md = (1/n)*sum(x*(yp-y))
    bd = (1/n)*sum(yp-y)
    mi = mi - learning_rate * md
    bi = bi - learning_rate * bd
    cost = (1/n) * sum([val**2 for val in (yp-y)])
            
print("slope =",mi," cost =",cost,"const =",bi)   
