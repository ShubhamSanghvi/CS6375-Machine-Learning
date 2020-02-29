import numpy as np
from cvxopt import matrix, solvers
from pdb import set_trace

class HardSVM:
    def fit(self,X,Y):
        n_feature = len(X[:,0])
        n_sample = Y.size
        n_paras = n_feature + 1
        # construct P
        P = np.zeros(n_paras)
        for i in range(n_feature):
            P[i]=1
        P = np.diag(P)

        # construct q
        q = np.zeros(n_paras)

        # construct G respect to constraint y(wx+b)>=1
        G = []
        for i in range(n_sample):
            tmp = np.zeros(n_paras)
            x_i = X[:,i]
            y_i = Y[0,i]
            tmp[0:n_feature] = y_i*x_i
            tmp[n_feature] = y_i
            G.append(tmp)

        G = np.array(G)

        # construct h
        h=np.zeros(n_sample)
        for i in range(n_sample):
            h[i] = 1

        # transform Gx >= h to Gx <= h as required
        G=-G; h=-h
        ret = solvers.qp(matrix(P),matrix(q),matrix(G),matrix(h))
        solution = ret['x']

        # decompose solution to w,b,ksi
        w = solution[0:n_feature]
        w = np.array(w).reshape(n_feature,1)
        b = solution[n_feature]
        self.W  = w
        self.b = b
        return self

    def predict(self,X):
        return np.dot(self.W.T,X)+self.b

    def evaluate(self,X,Y):
        O = self.predict(X)
        R = np.multiply(Y,O)
        n_right = len(R[R>0])
        accuracy = float(n_right)/Y.size
        return accuracy

def read_data(fname):
    fh = open(fname, encoding='utf-8-sig')
    contents = fh.readlines()
    fh.close()
    X=[];Y=[]
    for line in contents:
        sample = line.strip().split(',')
        X.append(sample[:-1])
        Y.append(sample[-1])
    X = np.array(X,dtype = 'float'); X = X.T
    Y = np.array(Y,dtype = 'float').reshape(1,-1)
    return X,Y

def phi(x):
    # this is the feature mapping funtion
    # I am using all degree 2 curve here, including polynomial, circle, elliptics ......
    xf = []
    for v in x:
        xf.append(v*v)
    n = x.size
    for i in range(n):
        for j in range(i+1,n):
            xf.append(x[i]*x[j])
    for v in x:
        xf.append(v)
    return xf

def main(args):
    datafile = '../mystery.data'
    # each column of X/Y is a feature_vec/label
    oriX,Y = read_data(datafile)
    X = np.array([phi(x) for x in oriX.T])
    X = X.T
    svm = HardSVM().fit(X,Y)
    # accuracy must be 1.0 cause we are learning perfect classifier
    print('The accuracy is {}'.format(svm.evaluate(X,Y)))
    print('Weight and bias is:\n {}\n {}'.format(svm.W.reshape(1,-1), svm.b))
    print('Margin is {}'.format(1.0/np.linalg.norm(svm.W)))

    # find the support vectors
    for i,x in enumerate(X.T):
        diff = abs(svm.predict(x)) -1
        assert(diff >= 0) # check if you result classifer is correct
        if  diff < 1e-5:
            print("Sample {} is support vector {}".format(i,oriX[:,i]))

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])