import numpy as np
from sys import argv
import math
from matplotlib import pyplot as plt

def random_ric(k,n):
	'''
		Initialized by ric
	'''
	prob =np.zeros((n,k))
	for i in range(n):
		a = np.random.random(3)
		a /= a.sum()
		prob[i]= a
	return prob

def gmm(ric,data,k,n,d,sigma):
	'''
		Gaussian Mixture Model and EM
	'''
	Miu,Pi,Sigma = None,None,sigma
	Ric = ric

	#use for log likeli-hood
	pre = -float("inf")
	likely = None

	while True:
		#E-step
		if Miu is not None and Pi is not None:
			Ric,likely = E_step(data,Miu,Sigma,Pi,k,n,d)

		#M-step
		Miu,Pi,Sigma = M_step(Ric, data, Sigma)


		#log likeli-hood
		if likely is not None:
			cur = sum(np.log(likely))  
			if cur-pre < 1e-15:
				break        
			pre = cur
	
	return Miu,Pi,Sigma,Ric
	

def E_step(data,miu,sigma,pi,k,n,d):

	x = np.mat(np.zeros([n, k]))
	N = np.mat(np.zeros([n, k]))

	for i in range(k):
		x = data-np.tile(miu[i, :],(n, 1)) 

		sigma_inv = np.linalg.inv(sigma[:, :, i])

		sigma_det = np.linalg.det(np.mat(sigma_inv))

		if sigma_det<0:
			sigma_det = 0

		for j in range(n):
			v = x[j]*sigma_inv*x[j].T
			N[j,i] = math.pow((2*(math.pi)),(-d/2)) * math.sqrt(sigma_det)* np.exp(-0.5*v)


	#ric calculation
	numerator = np.multiply(pi,N)

	denominator = np.mat(np.zeros([n,1]))

	for i in range(len(numerator)):
		denominator[i] = np.sum(numerator[i])

	new_ric = np.array(np.divide(numerator,denominator))

	return new_ric, denominator


def M_step(ric,data,sigma):

		Sigma = sigma

		Nk = sum(ric,0)

		#M-step
		Miu = np.mat(np.diag((1/Nk)))*(ric.T) * data

		Pi = Nk/n

		for i in range(k):
			x = data - np.tile(Miu[i], (n,1))
			Sigma[:,:,i] = (x.T * np.mat(np.diag(ric[:, i].T) * x)) / Nk[i]

		return Miu, Pi, Sigma

	

if __name__ == "__main__":
	_ , filename = argv

	fp = open(filename,'r')
	k = 3

	data = []

	p_data = []

	for line in fp.readlines() :
		line = line.rstrip().split(',')
		data.append([float(line[0]),float(line[1])])
		p_data.append([float(line[0]),float(line[1])])
	data = np.mat(data)
	n,d = np.shape(data)

	init_ric = random_ric(k,n)

	s = np.zeros((d,d,k))

	f_miu, f_pi, f_sigma, f_ric = gmm(init_ric,data,k,n,d,s)

	f_miu = np.array(f_miu)

	print "Miu: "+str(f_miu)+"\n"
	print "Pi: "+str(f_pi)+"\n"
	print "Sigma: "+str(f_sigma)+"\n"

	mark = []
	marks = ["or","ob","og"]
	for i in range(len(f_ric)):
		mark.append(np.argmax(f_ric[i]))

	for i in xrange(n):
		plt.plot(p_data[i][0], p_data[i][1], marks[mark[i]])  

	for i in range(k):  
		plt.plot(f_miu[i][0], f_miu[i][1], 'xk')  

	plt.show()  