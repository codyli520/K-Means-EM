import matplotlib.pyplot as plt
import math
import random as rand
import numpy as np
from sys import argv

   
def k_means(data,k_val,n,d,c):
	'''
	  Calculate clusters by k-means algorithm
	'''
	cluster_rec = [[] for x in xrange(n)]
	finished = False 
	
	while not finished:
		finished = True

		#calculate distance and assign centroid
		for i in xrange(n):
			min_dist, c_index = 100.0, 0.0
			for j in range(k):

				#calculate distance to centroid
				dist = math.sqrt(pow(c[j][0]-data[i][0],2) + pow(c[j][1]-data[i][1],2))
				if dist < min_dist:
					min_dist, c_index = dist, j

			if len(cluster_rec[i]) == 0 or cluster_rec[i][0] != c_index:
				cluster_rec[i] = [c_index, min_dist**2]
				finished = False

		#re-calculate centroid
		for j in xrange(k_val):
			subset = []
			for v in range(n):
				if cluster_rec[v][0] == j:
					subset.append(data[v])

					'''un-comment if want to plot'''
					if len(data[v]) < 3:
						data[v].append(j)
					elif data[v][2] != j:
						data[v][2] = j

			subset_m = np.mat(subset)
			temp = np.mean(subset_m, axis = 0)
			c[j] = np.array(temp).reshape(-1).tolist()
			
	return c 


if __name__ == "__main__":
	_ , filename = argv

	fp = open(filename,'r')
	k = 3
	x = []
	y = []
	data = []

	for line in fp.readlines() :
	    line = line.rstrip().split(',')
	    data.append([float(line[0]),float(line[1])])

	num, dim = len(data), 2 
	random_index = rand.sample(range(0,num),k)
	centroids = []

	for i in range(k):  
	    centroids.append(data[random_index[i]])  

	#print centroids
	f_centroid = k_means(data,k,num,dim,centroids)
	print f_centroid

	'''un-comment to plot'''
	mark = ['or', 'ob', 'og','ok']  
 
	for i in xrange(num):
		plt.plot(data[i][0], data[i][1], mark[data[i][2]])  


	for i in range(k):  
		plt.plot(f_centroid[i][0], f_centroid[i][1], 'xk', markersize = 12)  

	plt.show()  



