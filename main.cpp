#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include<string.h>
#include <sys/mman.h>
#include <sys/stat.h>
using namespace std;

struct Data {
    vector<double> features;
    int label;
    Data(vector<double> f, int l) : features(f), label(l)
    {}
};
struct Param {
    vector<double> wtSet;//weight
};


class LR {
public:
    void train();
    void predict();
    int loadModel();
    int storeModel();
    LR(string trainFile, string testFile, string predictOutFile);

private:
    vector<Data> trainDataSet;
    vector<Data> testDataSet;
    vector<int> predictVec;
    Param param;
    string trainFile;
    string testFile;
    string predictOutFile;
    string weightParamFile = "modelweight.txt";

private:
    bool init();
    bool loadTrainData();
    bool loadTestData();
    int storePredict(vector<int> &predict);
    void initParam();
    double wxbCalc(const Data &data);
    double sigmoidCalc(const double wxb);
    double lossCal();
    double gradientSlope(const vector<Data> &dataSet, int index, const vector<double> &sigmoidVec);

private:
    int featuresNum;
    const double wtInitV = 1.0;
    const double stepSize = 0.1;
    const int maxIterTimes = 10;
    const double predictTrueThresh = 0.6;
    const int train_show_step = 10;
};

LR::LR(string trainF, string testF, string predictOutF)
{
    trainFile = trainF;
    testFile = testF;
    predictOutFile = predictOutF;
    featuresNum = 0;
    init();
}

inline Data ParseTrainDataItem(char *str) {
   vector<double> v;

   char *begin = str, *end;
   errno       = 0;
   double tmp  = strtod(begin, &end);

   while (errno == 0 && end != begin) {
       v.push_back(tmp);
       if (*end == '\n')
           break;
       begin = end + 1;
       tmp   = strtod(begin, &end);
   }
   // error occured.
   if (v.size() == 0)
       return Data(v, -1);

   int ftf = (int)v.back();
   v.pop_back();
   return Data(v, ftf);
}
inline void handleError(const char *msg) {
   perror(msg);
   exit(255);
}
// Memory Mapping
///////////////////////////////////////////////////////////////////////////////
char *mapFile(const char *fname, size_t &length) {
   int fd = open(fname, O_RDONLY);
   if (fd == -1)
       handleError("open");

   // obtain file size
   struct stat sb;
   if (fstat(fd, &sb) == -1)
       handleError("fstat");

   length = sb.st_size;

   char *addr = static_cast<char *>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0u));
   if (addr == MAP_FAILED)
       handleError("mmap");

   // TODO close fd at some point in time, call munmap(...)
   return addr;
}
bool LR::loadTrainData()
{
    size_t length;
    char * f     = mapFile(trainFile.c_str(), length);
    char * l     = f + length;

    // the first line.
    auto parseResult = ParseTrainDataItem(f);
    if (parseResult.label != -1)
        trainDataSet.push_back(parseResult);

    while (f && f != l) {
        if ((f = static_cast<char *>(memchr(f, '\n', l - f)))) {
            parseResult = ParseTrainDataItem(f);
            if (parseResult.label != -1)
                trainDataSet.push_back(parseResult);
            f++;
        }
    }
    munmap(f, length);
    return true;
}

void LR::initParam()
{
    int i;
    for (i = 0; i < featuresNum; i++) {
        param.wtSet.push_back(wtInitV);
    }
}

bool LR::init()
{
    trainDataSet.clear();
    bool status = loadTrainData();
    if (status != true) {
        return false;
    }
    featuresNum = trainDataSet[0].features.size();
    param.wtSet.clear();
    initParam();
    return true;
}




inline double LR::sigmoidCalc(const double wxb)
{
    double expv = exp(-1 * wxb);
    double expvInv = 1 / (1 + expv);
    return expvInv;
}


double LR::lossCal()
{
    double lossV = 0.0L;
    int i;

    for (i = 0; i < trainDataSet.size(); i++) {
        lossV -= trainDataSet[i].label * log(sigmoidCalc(wxbCalc(trainDataSet[i])));
        lossV -= (1 - trainDataSet[i].label) * log(1 - sigmoidCalc(wxbCalc(trainDataSet[i])));
    }
    lossV /= trainDataSet.size();
    return lossV;
}


double LR::gradientSlope(const vector<Data> &dataSet, int index, const vector<double> &sigmoidVec)
{
    double gsV = 0.0L;
    int i;
    double sigv, label;
    for (i = 0; i < dataSet.size(); i++) {
        sigv = sigmoidVec[i];
        label = dataSet[i].label;
        gsV += (label - sigv) * (dataSet[i].features[index]);
    }

    gsV = gsV / dataSet.size();
    return gsV;
}

inline double LR::wxbCalc(const Data &data)
{
    double mulSum = 0.0L;
    int i;
    double wtv, feav;
    for (i = 0; i < param.wtSet.size(); i++) {
        wtv = param.wtSet[i];
        feav = data.features[i];
        mulSum += wtv * feav;
    }

    return mulSum;
}

void LR::train()
{
    double sigmoidVal;
    double wxbVal,alpha;
    int i, j;
    vector <int> data_index;
    int count=0;
    for (i = 0; i < maxIterTimes; i++) {
        for(int l=0;l<trainDataSet.size();l ++ )
             data_index.push_back(l);
        for (j = 0; j < trainDataSet.size(); j ++ ) {
            alpha = 2/ (1.0 + j + i) + 0.001;
            int rand_index=rand() % (data_index.size());
            wxbVal = wxbCalc(trainDataSet[data_index[rand_index]]);
            sigmoidVal = sigmoidCalc(wxbVal);
            double err=trainDataSet[data_index[rand_index]].label-sigmoidVal;
            
            for (int k = 0; k < param.wtSet.size(); k++) {
               
                    param.wtSet[k] += alpha * err * trainDataSet[data_index[rand_index]].features[k];
                    
                
           }
            int index_back=data_index.back();
            data_index[rand_index]=index_back;
            data_index.pop_back();
        }
        
    }
}

void LR::predict()
{
    double sigVal;
    int predictVal;

    loadTestData();
    for (int j = 0; j < testDataSet.size(); j++) {
        sigVal = sigmoidCalc(wxbCalc(testDataSet[j]));
        predictVal = sigVal >= predictTrueThresh ? 1 : 0;
        predictVec.push_back(predictVal);
    }
    
    storePredict(predictVec);
}

int LR::loadModel()
{
    string line;
    int i;
    vector<double> wtTmp;
    double dbt;

    ifstream fin(weightParamFile.c_str());
    if (!fin) {
        cout << "打开模型参数文件失败" << endl;
        exit(0);
    }
    
    getline(fin, line);
    stringstream sin(line);
    for (i = 0; i < featuresNum; i++) {
        char c = sin.peek();
        if (c == -1) {
            cout << "模型参数数量少于特征数量，退出" << endl;
            return -1;
        }
        sin >> dbt;
        wtTmp.push_back(dbt);
    }
    param.wtSet.swap(wtTmp);
    fin.close();
    return 0;
}

int LR::storeModel()
{
    string line;
    int i;

    ofstream fout(weightParamFile.c_str());
    if (!fout.is_open()) {
        cout << "打开模型参数文件失败" << endl;
    }
    if (param.wtSet.size() < featuresNum) {
        cout << "wtSet size is " << param.wtSet.size() << endl;
    }
    for (i = 0; i < featuresNum; i++) {
        fout << param.wtSet[i] << " ";
    }
    fout.close();
    return 0;
}


bool LR::loadTestData()
{
    ifstream infile(testFile.c_str());
    string lineTitle;

    if (!infile) {
        cout << "打开测试文件失败" << endl;
        exit(0);
    }
    
    while (infile) {
        vector<double> feature;
        string line;
        getline(infile, line);
        if (line.size() > featuresNum) {
            stringstream sin(line);
            double dataV;
            int i;
            char ch;
            i = 0;
            while (i < featuresNum && sin) {
                char c = sin.peek();
                if (int(c) != -1) {
                    sin >> dataV;
                    feature.push_back(dataV);
                    sin >> ch;
                    i++;
                } else {
                    cout << "测试文件数据格式不正确" << endl;
                    return false;
                }
            }
            testDataSet.push_back(Data(feature, 0));
        }
    }
    
    infile.close();
    return true;
}

bool loadAnswerData(string awFile, vector<int> &awVec)
{
    ifstream infile(awFile.c_str());
    if (!infile) {
        cout << "打开答案文件失败" << endl;
        exit(0);
    }

    while (infile) {
        string line;
        int aw;
        getline(infile, line);
        if (line.size() > 0) {
            stringstream sin(line);
            sin >> aw;
            awVec.push_back(aw);
        }
    }
    
    infile.close();
    return true;
}

int LR::storePredict(vector<int> &predict)
{
    string line;
    int i;

    ofstream fout(predictOutFile.c_str());
    if (!fout.is_open()) {
        cout << "打开预测结果文件失败" << endl;
    }
    for (i = 0; i < predict.size(); i++) {
        fout << predict[i] << endl;
    }
    fout.close();
    return 0;
}

int main(int argc, char *argv[])
{
    vector<int> answerVec;
//    /Users/pluto/huawei/code1/hu/hu
    vector<int> predictVec;
    int correctCount;
    double accurate;
    string trainFile = "/data/train_data.txt";
    string testFile = "/data/test_data.txt";
    string predictFile = "/projects/student/result.txt";

    string answerFile = "/projects/student/answer.txt";
    
    LR logist(trainFile, testFile, predictFile);
    
    cout << "ready to train model" << endl;
    logist.train();
    
    cout << "training ends, ready to store the model" << endl;
    logist.storeModel();

#ifdef TEST
    cout << "ready to load answer data" << endl;
    loadAnswerData(answerFile, answerVec);
#endif
    cout << "let's have a prediction test" << endl;
    logist.predict();

#ifdef TEST
    loadAnswerData(predictFile, predictVec);
    cout << "test data set size is " << predictVec.size() << endl;
    correctCount = 0;
    for (int j = 0; j < predictVec.size(); j++) {
        if (j < answerVec.size()) {
            if (answerVec[j] == predictVec[j]) {
                correctCount++;
            }
        } else {
            cout << "answer size less than the real predicted value" << endl;
        }
    }
    
    accurate = ((double)correctCount) / answerVec.size();
    cout << "the prediction accuracy is " << accurate << endl;
#endif

    return 0;
}

