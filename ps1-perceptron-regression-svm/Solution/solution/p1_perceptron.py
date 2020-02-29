#coding:utf-8
import math
import numpy as np

DATA_FILE = "../perceptron.data"
NUM_FEATURE = 4
EPS = 1e-5

def verify(X,Y,W,b):
	M = Y.size
	for i in range(M):
		x_i = X[:,i:i+1]
		y_i = Y[i]
		if y_i * F(x_i,W,b) <= 0 :
			return False
	return True

def step_size(t):
	return 1 # 46
	# return 0.1 #46
	# return 10 #46
	# return 1.0/(100+math.sqrt(t)) #187
	# return 1.0/(10+t) #95
	# return 1.0/(5+t) #387
	# return 1.0/(3+t) # 2171
	# return 1.0/(3.5+0.5*t) # 178


def F(x,w,b):
	return np.dot(w.T,x)+b

def calculate_grad(X,Y,W,b):
	M = Y.size
	w_grad = np.zeros((NUM_FEATURE,1)); b_grad =0
	for i in range(M):
		x_i = X[:,i:i+1]
		y_i = Y[i]
		# incorrect classification
		if y_i * F(x_i,W,b) <= 0 :
			w_grad += -y_i*x_i
			b_grad += -y_i
	return (w_grad,b_grad)

def calculate_grad_stochastic(X,Y,W,b,i):
	M = Y.size
	w_grad = np.zeros((NUM_FEATURE,1)); b_grad = 0
	# use i element to approximate gradient
	x_i = X[:,i:i+1]
	y_i = Y[i]
	if y_i * F(x_i,W,b) <= 0:
		w_grad += -y_i*x_i
		b_grad += -y_i
	return (w_grad,b_grad)

def main_1(X,Y):
	# training initialize
	t = 0
	W = np.zeros((NUM_FEATURE,1),dtype='float'); b = 0
	cur_grad= [np.ones((NUM_FEATURE,1),dtype='float'),1]
	# start
	while np.linalg.norm(cur_grad[0]) > EPS or np.linalg.norm(cur_grad[1]) > EPS:
		t += 1
		cur_grad = calculate_grad(X,Y,W,b)
		W = W - cur_grad[0]*step_size(t)
		b = b - cur_grad[1]*step_size(t)
		if t<4:
			print('Iter',t,'with value',W.reshape((1,NUM_FEATURE)),b)

	print('The total number of iterations is',t)
	print('The Final Parameter is',W.reshape((1,NUM_FEATURE)),b)


def main_2(X,Y):
	# training initialize
	t = 0 ;	n_it = 0
	W = np.zeros((NUM_FEATURE,1),dtype='float'); b = 0
	cur_grad= [np.zeros((NUM_FEATURE,1),dtype='float'),1]
	while (True):
		cur_grad = calculate_grad_stochastic(X,Y,W,b,t)
		W = W - cur_grad[0]*step_size(n_it)
		b = b - cur_grad[1]*step_size(n_it)
		t+=1 ; n_it+=1
		if t == Y.size:
			t = 0
			if verify(X,Y,W,b):
				print('The total number of iterations is',n_it)
				print('The Final Parameter is',W.reshape((1,NUM_FEATURE)),b)
				return
		if n_it<4:
			print('Iter',t,'with value',W.reshape((1,NUM_FEATURE)),b)


def read_data(fname):
	fh = open(fname)
	contents = fh.readlines()
	fh.close()
	X=[];Y=[]
	for line in contents:
		sample = line.strip().split(',')
		X.append(sample[:-1])
		Y.append(sample[-1])
	X = np.array(X,dtype = 'float');X=X.T
	Y = np.array(Y,dtype = 'float')
	return X,Y


if __name__ == '__main__':
	X,Y = read_data(DATA_FILE)
	main_1(X,Y)
	print('\n\n')
	main_2(X,Y)