from sklearn import datasets
import matplotlib.pyplot as plt
import sklearn
import numpy as np
def predict_prob(X,theta):
     return 1/(1+np.exp(-1*(np.dot(X,theta))))
iris = sklearn.datasets.load_iris()
X = iris.data[:,:2]
X=(X-X.mean())/X.std()
y = (iris.target != 0) * 1

plt.figure(figsize=(10, 6))
plt.scatter(X[y==0][:,0],X[y==0][:,1],color='b',label=0)
plt.scatter(X[y==1][:,0],X[y==1][:,1],color='r',label=1)
plt.legend()
plt.show()
ite=2300;
a=0.3;
intercept = np.ones((X.shape[0], 1))
X=np.concatenate((intercept, X), axis=1)
theta=np.zeros(X.shape[1])
cost=np.zeros([ite])
for i in range(ite):
    z=np.dot(X,theta)
    h=1/(1+np.exp(-1*z)) 
    gradient=np.dot(X.T,(h-y))/y.size
    theta=theta-a*gradient
    cost[i]=(-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
print(theta)
print(cost)
p=input("enter the value of x1")
q=input("enter the value of x2 ")
p=float(p)
q=float(q)
x=np.array([1,p,q])
theta2=np.array([theta[0],theta[1],theta[2]])
theta2=np.transpose(theta2)
c=np.dot(x,theta2)
t=1/(1+np.exp(-1*c)) 
print(t)
0


