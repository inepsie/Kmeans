import numpy as np
import random as rnd
import sys
import matplotlib.pyplot as plt
import math
import cv2
import glob
from datetime import datetime
import os
import shutil


class DataSet:

    def __init__(self):
        self.centre1=np.array([15,15])           # centre du premier cluster
        self.centre2=np.array([0,0])           # centre du second cluster
        self.centre3=np.array([-5,5]) 
        self.centre4=np.array([5,5])           #
        self.sigma1=np.array([[2,2],[2,2]]) # matrice de covariance du premier cluster
        self.sigma2=np.array([[3.5,2],[2,3.5]]) # matrice de covariance du second cluster
        self.sigma3=np.array([[4.5,2],[2,4.5]])
        self.sigma4=np.array([[6,2],[2,3.5]])
        self.taille1=50                        # nombre de points du premier cluster
        self.taille2=400                        # nombre de points du second cluster
        self.taille3=200
        self.taille4=1000
        self.cluster1=np.random.multivariate_normal(self.centre1,self.sigma1,self.taille1)
        self.cluster2=np.random.multivariate_normal(self.centre2,self.sigma2,self.taille2)
        self.cluster3=np.random.multivariate_normal(self.centre3,self.sigma3,self.taille3)
        self.cluster4=np.random.multivariate_normal(self.centre4,self.sigma4,self.taille4)

class Kmeans:

    def __init__(self, dataset, nb_class, pow_minkowski):
        self.dataset = dataset
        self.dataset_length = len(dataset)
        self.dim = len(dataset[0])
        self.nb_class = nb_class
        self.centers = []
        self.nb_steps = 0
        self.pow_minkowski = pow_minkowski
        self.current_inertie = 0
        self.history_inertie = []

        self.dataset = np.insert(self.dataset, self.dim, 0, axis=1)

    def rndInitCenters(self):
        pos = []
        while(len(self.centers)<self.nb_class):
            r = rnd.randint(0,self.dataset_length-1)
            if r not in pos:
                pos.append(r)
                self.centers.append(self.dataset[r])

    def giveAClass(self):
        self.current_inertie = 0
        for elmt in self.dataset:
            elmt[-1] = self.findNearest(elmt)
        self.history_inertie.append(self.current_inertie)


    def findNearest(self, elmt):
        nearest = 0
        distance_min = 999999999
        for c in range(len(self.centers)):
            distance = 0
            for i in range(self.dim):
                distance += pow(abs(elmt[i]-self.centers[c][i]), self.pow_minkowski)
            distance = pow(distance, 1/self.pow_minkowski)
            if distance<distance_min:
                distance_min = distance
                nearest = c
        self.current_inertie += pow(distance_min, 2)
        return nearest

    def newCenters(self):
        for c in range(len(self.centers)):
            sum = np.zeros(self.dim)
            n = 0
            for elmt in self.dataset:
                if elmt[-1]==c:
                    n+=1
                    for i in range(self.dim):
                        sum[i] += elmt[i]
            for i in range(self.dim):
                sum[i] = sum[i]/n
            self.centers[c] = sum

    def start(self):
        self.rndInitCenters()
        for i in range(1000):
            self.giveAClass()

            plt.scatter([elmt[0] for elmt in self.dataset if elmt[-1]==0], [elmt[1] for elmt in self.dataset if elmt[-1]==0], color="pink")
            plt.scatter([elmt[0] for elmt in self.dataset if elmt[-1]==1], [elmt[1] for elmt in self.dataset if elmt[-1]==1], color="blue")
            plt.scatter([elmt[0] for elmt in self.dataset if elmt[-1]==2], [elmt[1] for elmt in self.dataset if elmt[-1]==2], color="green")
            plt.scatter([elmt[0] for elmt in self.dataset if elmt[-1]==3], [elmt[1] for elmt in self.dataset if elmt[-1]==3], color="yellow")
            plt.scatter([elmt[0] for elmt in self.dataset if elmt[-1]==4], [elmt[1] for elmt in self.dataset if elmt[-1]==4], color="black")
            plt.scatter([elmt[0] for elmt in self.dataset if elmt[-1]==5], [elmt[1] for elmt in self.dataset if elmt[-1]==5], color="brown")
            plt.title("K-means (minkowski="+str(self.pow_minkowski)+")")
            plt.savefig("./minkowski-"+str(self.pow_minkowski)+"/"+str(datetime.now())+".png")

            if(len(self.history_inertie)>10 and 
            self.history_inertie[-1]==self.history_inertie[-2] and 
            self.history_inertie[-2]==self.history_inertie[-3] and
            self.history_inertie[-3]==self.history_inertie[-4] and
            self.history_inertie[-4]==self.history_inertie[-5]):
                break

            self.newCenters()
            self.nb_steps += 1

#lance des kmeans avec des calculs de distance de minkowski différents
def kmeans_launcher(nb_class, min, max, data):
    if(max<min):
        print("L'argument \"min\" de kmeans_launcher doit etre inferieur à l'argument \"max\"")
        return

    f = list(glob.glob("./*minkowski*"))
    for elmt in f:
        if os.path.isdir(elmt):
            shutil.rmtree(elmt)
        else:
            os.remove(elmt)

    for i in range(min, max+1):
        path = "./minkowski-"+str(i)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        kmeans = Kmeans(data, nb_class, i)
        kmeans.start()
        #print(kmeans.history_inertie)
        img_array = []
        images = list(glob.glob(path+"/*.png"))
        images.sort()
        for elem in images:
            img = cv2.imread(elem)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        out1 = cv2.VideoWriter(path+"/kmeans.avi-minkowski-"+str(i)+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
        out2 = cv2.VideoWriter("./kmeans.avi-minkowski-"+str(i)+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
        for i in range(len(img_array)):
            out1.write(img_array[i])
            out2.write(img_array[i])
        out1.release()
        out2.release()
        plt.close()
        plt.plot([x for x in range(len(kmeans.history_inertie))], kmeans.history_inertie)
        plt.title("Evolution de l'inertie (minkowski="+str(kmeans.pow_minkowski)+")")
        plt.savefig("./inertie-minkowski-"+str(kmeans.pow_minkowski)+".png")
        plt.close()

def main(**kwargs):
    dataset = DataSet()
    d = np.concatenate((dataset.cluster1, dataset.cluster2, dataset.cluster3, dataset.cluster4), axis=0)
    kmeans_launcher(nb_class=5, min=2, max=2, data=d)

main()
