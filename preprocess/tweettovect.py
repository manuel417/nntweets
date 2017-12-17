import tensorflow as tf
import csv
import math

class TweetToVect:
    def __init__(self, max_tweet):
        self.max_tweet = max_tweet

    def add_padding(self, data):
        if data:
            L = len(data)
            difference = self.max_tweet - L
            for i in range (0, difference):
                data.append(' ')
            return data
        else:
            return data

    def convert(self, tweet_string):
        result = []
        for c in tweet_string:
            result.append(ord(c))

        return self.add_padding(result)

class CSVTweetsToMatrix:
    def __init__(self, inputname, outputname, max_tweet):
        self.inputname = inputname
        self.outputname = outputname
        self.max_tweet = max_tweet

    def convert(self):
        T = TweetToVect(self.max_tweet)
        with open(self.inputname, "rt") as input_csv:
            next(input_csv)
            csv_reader = csv.reader(input_csv, delimiter=',')
            output_file = open(self.outputname, "wt")
            output_csv = csv.writer(output_file, delimiter = ",")
            i = 0
            for row in csv_reader:
                label = row[0]
                tweet = row[1]
                vect_tweet = T.convert(tweet_string=tweet)
                labeled_data = vect_tweet + [label]
                output_csv.writerow(labeled_data)
            input_csv.close()
            output_file.close()


class GenerateTrainDevTestSets:
    def __init__(self, filename, numExamples, split=[0.60, 0.20, 0.20]):
        self.filename = filename
        self.split = split
        self.numExamples=numExamples

    def getRecordsSplits(self):
        train = self.split[0]
        dev = self.split[1]
        test = self.split[2]

        trainExamples = math.floor(self.numExamples * train)
        devExamples = math.floor(self.numExamples * dev)
        testExamples = math.floor(self.numExamples * test)

        while ((trainExamples + devExamples + testExamples) < self.numExamples):
            trainExamples = trainExamples + 1
        return (trainExamples, devExamples, testExamples)

    def generate(self):
        training = "traning.csv"
        dev = "dev.csv"
        test = "test.csv"
        recordSplit = self.getRecordsSplits()
        print(recordSplit)
        Ntrain = recordSplit[0]
        Ndev = recordSplit[1]
        Ntest = recordSplit[2]
        with open(self.filename, "rt") as input_file:
            train_file = open(training, "wt")
            for i in range(0, Ntrain):
                nextLine = input_file.readline()
                train_file.write(nextLine)
            train_file.close()
            dev_file = open(dev, "wt")
            for i in range(0, Ndev):
                nextLine = input_file.readline()
                dev_file.write(nextLine)
            dev_file.close()
            test_file = open(test, "wt")
            for i in range(0, Ntest):
                nextLine = input_file.readline()
                test_file.write(nextLine)
            test_file.close()






