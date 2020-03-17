import math
import datetime
import sys
import numpy as np


class LR:
    def __init__(self, train_file_name, test_file_name, predict_result_file_name):
        self.train_file = train_file_name
        self.predict_file = test_file_name
        self.predict_result_file = predict_result_file_name
        self.max_iters = 30
        self.rate = 0.1
        self.feats = []
        self.labels = []
        self.feats_test = []
        self.labels_predict = []
        self.param_num = 0
        self.error=0.0
        self.weight = []

    def loadDataSet(self, file_name, label_existed_flag):
        feats = []
        labels = []
        fr = open(file_name)
        lines = fr.readlines()
        for line in lines:
            temp = []
            allInfo = line.strip().split(',')
            dims = len(allInfo)
            if label_existed_flag == 1:
                for index in range(dims-1):
                    temp.append(float(allInfo[index]))
                feats.append(temp)
                labels.append(float(allInfo[dims-1]))
            else:
                for index in range(dims):
                    temp.append(float(allInfo[index]))
                feats.append(temp)
        fr.close()
        feats = np.array(feats)
        labels = np.array(labels)
        return feats, labels

    def loadTrainData(self):
        self.feats, self.labels = self.loadDataSet(self.train_file, 1)

    def loadTestData(self):
        self.feats_test, self.labels_predict = self.loadDataSet(
            self.predict_file, 0)

    def savePredictResult(self,predictlist):
        f = open(self.predict_result_file, 'w')
        for i in range(len(predictlist)):
            f.write(str(predictlist[i])+"\n")
        f.close()

    def sigmod(self, x):
        return 1.0/(1+np.exp(-x))
        
        

    def printInfo(self):
        print(self.train_file)
        print(self.predict_file)
        print(self.predict_result_file)
        print(self.feats)
        print(self.labels)
        print(self.feats_test)
        print(self.labels_predict)

    def initParams(self):
        self.weight = np.ones((self.param_num,), dtype=np.float)

    def compute(self, param_num, feats, w):
        return self.sigmod(np.sum(feats * w))

    def error_rate(self, recNum, label, preval):
        return label - preval
    def classify_vector(self,in_x, weights):
        prob = self.compute(self.param_num,in_x,weights)
        if prob > 0.5:
            return 1.0
        return 0.0
    def predict(self):
        self.loadTestData()
        truepredict=[]
        for i in range(len(self.feats_test)):
            preval = self.classify_vector(self.feats_test[i], self.weight)
            truepredict.append(int(preval))
        self.savePredictResult(truepredict)

    def train(self):
        self.loadTrainData()
        recNum = len(self.feats)
        self.param_num = len(self.feats[0])
        self.initParams()
        # ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S,f'
        for j in range(self.max_iters):
            data_index = list(range(recNum))
            for i in range(recNum):
                alpha = 4 / (1.0 + j + i) + 0.01
                rand_index = int(np.random.uniform(0, len(data_index)))
                h =self.compute(self.param_num,self.feats[data_index[rand_index]],self.weight)
                err = self.labels[data_index[rand_index]]-h
                self.weight = self.weight + alpha * err * self.feats[data_index[rand_index]]
                del(data_index[rand_index])

def print_help_and_exit():
    print("usage:python3 main.py train_data.txt test_data.txt predict.txt [debug]")
    sys.exit(-1)


def parse_args():
    debug = False
    if len(sys.argv) == 2:
        if sys.argv[1] == 'debug':
            print("test mode")
            debug = True
        else:
            print_help_and_exit()
    return debug


if __name__ == "__main__":
    debug = parse_args()
    train_file =  "/data/train_data.txt"
    test_file = "/data/test_data.txt"
    predict_file = "/projects/student/result.txt"
    lr = LR(train_file, test_file, predict_file)
    lr.train()
    lr.predict()

    if debug:
        answer_file ="/projects/student/answer.txt"
        f_a = open(answer_file, 'r')
        f_p = open(predict_file, 'r')
        a = []
        p = []
        lines = f_a.readlines()
        for line in lines:
            a.append(int(float(line.strip())))
        f_a.close()

        lines = f_p.readlines()
        for line in lines:
            p.append(int(float(line.strip())))
        f_p.close()

        print("answer lines:%d" % (len(a)))
        print("predict lines:%d" % (len(p)))

        errline = 0
        for i in range(len(a)):
            if a[i] != p[i]:
                errline += 1

        accuracy = (len(a)-errline)/len(a)
        print("accuracy:%f" %(accuracy))
