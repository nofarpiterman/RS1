import pandas as pa
from pandas import *
import numpy as np
import datetime
import os.path
import math
import sys

class SVD():

    def __init__(self, bu, bi, P, Q, ratings_train, ratings_validation, ratings_test, numIterations, lamda, gama, k):
        self.bu = bu
        self.bi = bi
        self.P = P
        self.Q = Q
        self.ratings_train = ratings_train
        self.ratings_validation = ratings_validation
        self.ratings_test = ratings_test
        self.mu = 0
        self.numIterations = numIterations
        self.lamda = lamda
        self.gama = gama
        self.k = k
        self.lasterror = 10000

    def initMainParameter(self, usersNum, itemsNum):
        self.bu = np.random.random_sample(usersNum)
        self.bi = np.random.random_sample(itemsNum)
        self.P = np.random.random_sample((usersNum, self.k))
        self.Q = np.random.random_sample((itemsNum, self.k))

    def RMSE(self, dataTest):
        n = 0
        errorS2 = 0
        pred = 0
        for i, j, r in dataTest:
            pred = self.getPredictionRating(i, j)
            # print ("i= "+str(i) +" |j= "+str(j)+ " |pred= "+str(pred))
            errorS2 = errorS2 + pow((r - pred), 2)
            n = n + 1
        if errorS2 is None or n == 0:
            return -1
        RMSE = np.sqrt(errorS2 / (n + 1))
        return RMSE

    def getPredictionRating(self, i, j):
        prediction = self.mu + self.bu[i] + self.bi[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def calcPredictionMatrix(self):
        return self.mu + self.bu[:, np.newaxis] + self.bi[np.newaxis, :] + self.P.dot(self.Q.T)

    def calcPredictionVector(self, i):
        return self.mu + self.bu[i, np.newaxis] + self.bi[np.newaxis, :] + self.P[i, :].dot(self.Q.T)

    def correct_param(self, i, j, r):
        pred = self.getPredictionRating(i, j)
        e = np.round(r - pred, 10)
        if math.isnan(pred):
            writeToFile2(0, 'backupsvd/svd_nan', "nan")
            writeToFile2(1, 'backupsvd/svd_nan', "mu=" + str(self.mu) + " | bu= " + str(self.bu[i]) + " | bi= " + str(
                self.bi[j]) + " | Pi= " + str(self.P[i, :]) + " | Qj= " + str(self.Q[j, :]))
            sys.exit()

        bui = self.bu[i].round(decimals=10)
        bij = self.bi[j].round(decimals=10)
        Qj = self.Q[j, :].round(decimals=10)
        Pi = self.P[i, :].round(decimals=10)

        self.bu[i] = bui + self.gama * (e - self.lamda * bui)
        self.bi[j] = bij + self.gama * (e - self.lamda * bij)

        self.Q[j, :] = Qj + self.gama * (e * Pi - self.lamda * Qj)
        self.P[i, :] = Pi + self.gama * (e * Qj - self.lamda * Pi)

    def trainOnce(self):
        x = len(self.ratings_train)
        z = 0
        for i, j, r in self.ratings_train:
            self.correct_param(i, j, r)

            print (str(round(z * 100 / x, 2)) + '%')
            z = z + 1

    def train(self):
        print (datetime.datetime.utcnow())
        path = "ratingsTrain.csv"
        # path = "ratingsTest.csv"
        # input
        pathP = 'backupsvd/P' + str(self.lamda) + '_' + str(self.gama) + '.csv'
        pathQ = 'backupsvd/Q' + str(self.lamda) + '_' + str(self.gama) + '.csv'
        pathBu = 'backupsvd/bu' + str(self.lamda) + '_' + str(self.gama) + '.csv'
        pathBi = 'backupsvd/bi' + str(self.lamda) + '_' + str(self.gama) + '.csv'
        last_error = 100000
        error = 10000
        threshold = 0.01
        index = 0
        trainSize = 0.7
        writeToFile(0, "start loadRating")
        self.ratings_train, self.ratings_validation, usersNum, itemsNum, self.mu = loadRating(path, trainSize)
        writeToFile(1, "start initMainParameter")

        # if os.path.isfile(pathP) and os.path.isfile(pathQ) and os.path.isfile(pathBu) and os.path.isfile(pathBi):
        self.loadTrain('P.csv', 'Q.csv', 'bu.csv', 'bi.csv')
        # else:
        # self.initMainParameter(usersNum, itemsNum)

        writeToFile(1, "finish initMainParameter")
        while error < last_error and error > threshold and index <= self.numIterations:
            writeToFile(1, "start trainOnce " + str(index))
            pa.DataFrame(self.P).to_csv(pathP, index=False)
            pa.DataFrame(self.Q).to_csv(pathQ, index=False)
            pa.DataFrame(self.bu).to_csv(pathBu, index=False)
            pa.DataFrame(self.bi).to_csv(pathBi, index=False)
            self.trainOnce()
            writeToFile(1, "start calcPredictionMatrix")
            last_error = error
            self.lasterror = last_error
            error = self.RMSE(self.ratings_validation)
            index = index + 1
            writeToFile(1, "rmse= " + str(error))
            print (datetime.datetime.utcnow())

    def loadTrain(self, pathP, pathQ, pathBu, pathBi):
        self.P = np.array(pa.read_csv(pathP))
        self.Q = np.array(pa.read_csv(pathQ))
        self.bu = np.array(pa.read_csv(pathBu).T)[0]
        self.bi = np.array(pa.read_csv(pathBi).T)[0]


class Content():

    def readMovies(self):
        data = pa.read_csv("movies.csv")
        data2 = data[['movieId', 'title', 'genres']]
        allMovies = data2.to_records(index=False)
        listLen = data2['movieId'].max() + 1
        for i in range(1, listLen+1):
            empty = []
            self.movies.insert(i, empty)
        for ((m, t, g)) in allMovies:
            g_n = g.split("|")
            for i in g_n:
                self.movies[m].append(i)
        print self.movies

    def __init__(self, userID):
        self.Genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
                       "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
                       "(no genres listed)"]
        self.userGenres = {}
        self.movies = []
        self.user = userID
        self.readMovies()
        #initial user genres vector
        for g in self.Genres:
            self.userGenres[g] = 0




    def itemOfUser(self, rating, u):
        userItemRating = []
        n = 0
        sum = 0
        for i, j, r in rating:
            if i == u:
                userItemRating.append((i, j, r))
                sum = sum + r
                n = n + 1
        if not userItemRating:
            return userItemRating, -1
        else:
            return userItemRating, sum / n

    def testData(self, path):
        data = pa.read_csv(path)
        data2 = data[['userId', 'movieId', 'rating']]
        rating_test = data2.to_records(index=False)
        return rating_test

    def writeToFile(self, new, text):
        if new == 0:
            file = open("errors.txt", "w")
        else:
            file = open("errors.txt", "a")
        file.write(text + '                ' + str(datetime.datetime.now()) + "\n")
        file.write("\n")
        file.flush()
        file.close()

    def dotproduct(self, v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(self, v):
        return math.sqrt(self.dotproduct(v, v))

    def angle(self, v1, v2):
        return math.acos(self.dotproduct(v1, v2) / (self.length(v1) * self.length(v2)))

    def aboveAVG(self, rating, u):
        itemAboveAVG = []
        userItemRating, avg = self.itemOfUser(rating, u)
        for i, j, r in userItemRating:
            if r > avg:
                itemAboveAVG.append((i, j, r))
        return itemAboveAVG

    def start(self, numRecommendatios):
        rating = []
        items = self.aboveAVG(self, rating, self.user)
        # j-
        #add the rating of the movie to every genre of it in the user vector
        for (i, j, r) in items:
            for g in self.movies[j]:
                self.userGenres[g] += r


def loadRating(path, trainSize):
    data = pa.read_csv(path)
    data2 = data[['userId', 'movieId', 'rating']]
    usersNum = data2['userId'].max() + 1
    itemsNum = data2['movieId'].max() + 1
    data_train = data2.sample(frac=trainSize)
    data_validation = data2.drop(data_train.index)
    ratings_train = data_train.to_records(index=False)
    ratings_validation = data_validation.to_records(index=False)
    mu = data2['rating'].mean()
    return ratings_train, ratings_validation, usersNum, itemsNum, mu


def writeToFile(new, text):
    if new == 0:
        file = open("errors.txt", "w")
    else:
        file = open("errors.txt", "a")
    file.write(text + '                ' + str(datetime.datetime.utcnow()) + "\n")
    file.write("\n")
    file.flush()
    file.close()


def writeToFile2(new, filename, text):
    filename = filename + '.txt'
    if new == 0:
        file = open(filename, "w")
    else:
        file = open(filename, "a")
    file.write(text + '                ' + str(datetime.datetime.utcnow()) + "\n")
    file.write("\n")
    file.flush()
    file.close()

    """
    gama learning rates 0.005
    lamda regularizations 0.02
    0.05
    """


"""def main():
    bu = []
    bi = []
    P = []
    Q = []
    ratings_train = []
    ratings_validation = []
    ratings_test = []
    pathP = 'P.csv'
    pathQ = 'Q.csv'
    pathBu = 'bu.csv'
    pathBi = 'bi.csv'
    numIterations = 100
    lamda = 0.02
    gama = 0.005
    k = 2  # latent features
    minGama = 0.02
    minLamda = 0.005
    minError = 10000000

    # svd = SVD(bu, bi, P, Q, ratings_train, ratings_validation, ratings_test, numIterations, lamda, gama, k)
    # svd.train()
    # svd.loadTrain(pathP,pathQ,pathBu,pathBi)
    i = 1
    lm = 0.02
    ga = 0.005
    writeToFile2(0, 'backupsvd/svd', "start")
    for x in range(1, 11):
        lm = x * 0.02
        for y in range(1, 21):
            ga = y * 0.005
            svd = SVD(bu, bi, P, Q, ratings_train, ratings_validation, ratings_test, numIterations, lm, ga, k)
            svd.train()
            if svd.lasterror < minError:
                minGama = svd.gama
                minLamda = svd.lamda
                minError = svd.lasterror
                writeToFile2(1, 'backupsvd/svd', str(i) + "minGama= " + str(minGama) + " |minLamda= " + str(
                    minLamda) + " |minError= " + str(minError))
            else:
                writeToFile2(1, 'backupsvd/svd', str(i) + "--NOT---Gama= " + str(svd.gama) + " |minLamda= " + str(
                    svd.lamda) + " |minError= " + str(svd.lasterror))
            i = i + 1
"""

def main():
    userId = 1
    content = Content(userId)
    content.readMovies()



if __name__ == '__main__':
    main()
