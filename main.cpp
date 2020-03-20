#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
// for mmap:
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
using namespace std;
// for timing:
// #include <chrono>
// using namespace chrono;

// #define LOG
// #define TEST

// Data Struct
///////////////////////////////////////////////////////////////////////////////
struct Data {
    vector<double> features;
    int            label;
    Data(vector<double> f, int l) : features(f), label(l) {}
};
struct Param {
    vector<double> wtSet;
};

// Utils
///////////////////////////////////////////////////////////////////////////////
inline void Log(const char *msg) {
#ifdef LOG
    printf("%s", msg);
#endif
}

inline void Log(double d) {
#ifdef LOG
    printf("%lf", d);
#endif
}

inline void handleError(const char *msg) {
    perror(msg);
    exit(255);
}

inline Data ParseDataItem(char *str, bool isTrainData) {
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

    int ftf = 0;
    if (isTrainData) {
        ftf = (int)v.back();
        v.pop_back();
    }
    return Data(v, ftf);
}

inline void loadAnswerData(string awFile, vector<int> &awVec) {
    ifstream infile(awFile.c_str());
    if (!infile)
        handleError("打开答案文件失败");
    while (infile) {
        string line;
        int    aw;
        getline(infile, line);
        if (line.size() > 0) {
            stringstream sin(line);
            sin >> aw;
            awVec.push_back(aw);
        }
    }
    infile.close();
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

    // madvise
    if (madvise(addr, length, MADV_WILLNEED | MADV_SEQUENTIAL) == -1)
        handleError("madvise error");

    // close fd at some point in time, call munmap(...)
    return addr;
}

// Learning
///////////////////////////////////////////////////////////////////////////////
string trainFile   = "/data/train_data.txt";
string testFile    = "/data/test_data.txt";
string predictFile = "/projects/student/result.txt";
string answerFile  = "/projects/student/answer.txt";

class LR {
  public:
    void train();
    void predict();
    LR();

  private:
    vector<Data> trainDataSet;
    vector<Data> testDataSet;
    vector<int>  predictVec;
    Param        param;

  private:
    void   loadTrainData();
    void   loadTestData();
    void   storePredict(vector<int> &predict);
    double wxb(const Data &data);
    double sigmoid(const double wxb);
    double lossCal();

  private:
    int          featuresNum       = 0;
    const double wtInitV           = 1.0;
    const double stepSize          = 0.1;
    const int    maxIterTimes      = 10;
    const double predictTrueThresh = 0.5;
    const int    train_show_step   = 10;
};

LR::LR() {
    loadTrainData();
    featuresNum = trainDataSet[0].features.size();

    // initParam
    for (int i = 0; i < featuresNum; i++) {
        param.wtSet.push_back(wtInitV);
    }
}

void LR::loadTrainData() {
    size_t length;
    char * f = mapFile(trainFile.c_str(), length);
    char * l = f + length;

    // the first line.
    auto parseResult = ParseDataItem(f, true);
    if (parseResult.label != -1)
        trainDataSet.push_back(parseResult);
    int i=0;
    while (i<10000&&f && f != l) {
        if ((f = static_cast<char *>(memchr(f, '\n', l - f)))) {
            parseResult = ParseDataItem(f, true);
            if (parseResult.label != -1)
                trainDataSet.push_back(parseResult);
            f++;
        }
        i++;
    }
    // munmap(f, length);
}

inline double LR::sigmoid(const double wxb) {
    double expv    = exp(-1 * wxb);
    double expvInv = 1 / (1 + expv);
    return expvInv;
}

// double LR::lossCal() {
//     double lossV = 0.0L;
//     for (int i = 0; i < trainDataSet.size(); i++) {
//         lossV -= trainDataSet[i].label * log(sigmoid(wxb(trainDataSet[i])));
//         lossV -= (1 - trainDataSet[i].label) * log(1 - sigmoid(wxb(trainDataSet[i])));
//     }
//     lossV /= trainDataSet.size();
//     return lossV;
// }

inline double LR::wxb(const Data &data) {
    double mulSum = 0.0L;
    for (int i = 0; i < param.wtSet.size(); i++) {
        mulSum += param.wtSet[i] * data.features[i];
    }
    return mulSum;
}

void LR::train() {
    double      wxbVal, alpha, err;
    int         randIndex;
    vector<int> index;
    int         count = 0;

    for (int i = 0; i < maxIterTimes; i++) {

        for (int k = 0; k < trainDataSet.size(); k++)
            index.push_back(k);

        for (int j = 0; j < trainDataSet.size(); j++) {
            alpha     = 1 / (1.0 + j + i) + 0.01;
            randIndex = rand() % (index.size());
            wxbVal    = wxb(trainDataSet[index[randIndex]]);
            err       = trainDataSet[index[randIndex]].label - sigmoid(wxbVal);

            for (int k = 0; k < param.wtSet.size(); k++) {
                param.wtSet[k] += alpha * err * trainDataSet[index[randIndex]].features[k];
            }
            int index_back   = index.back();
            index[randIndex] = index_back;
            index.pop_back();
        }
    }
}

void LR::predict() {
    double sigVal;
    int    predictVal;
    loadTestData();
    for (int j = 0; j < testDataSet.size(); j++) {
        sigVal     = sigmoid(wxb(testDataSet[j]));
        predictVal = sigVal >= predictTrueThresh ? 1 : 0;
        predictVec.push_back(predictVal);
    }
    storePredict(predictVec);
}

void LR::loadTestData() {
    size_t length;
    char * f = mapFile(testFile.c_str(), length);
    char * l = f + length;

    // the first line.
    auto parseResult = ParseDataItem(f, false);
    if (parseResult.label != -1)
        testDataSet.push_back(parseResult);
    
    while (f && f != l) {
        if ((f = static_cast<char *>(memchr(f, '\n', l - f)))) {
            parseResult = ParseDataItem(f, false);
            if (parseResult.label != -1)
                testDataSet.push_back(parseResult);
            f++;
        }
       
    }
    // munmap(f, length);
}

void LR::storePredict(vector<int> &predict) {
    char *out = new char[predict.size() * 2];
    for (int i = 0; i < predict.size(); i++) {
        out[2 * i]     = '0' + predict[i];
        out[2 * i + 1] = '\n';
    }

    FILE *file = fopen(predictFile.c_str(), "wb");
    if (file == NULL)
        handleError("打开预测结果文件失败");
    fwrite(out, predict.size() * 2, 1, file);
    fclose(file);
}

int main(int argc, char *argv[]) {
    LR logist;

    Log("training model.\n");
    logist.train();
    Log("training ends.\n");

#ifdef TEST
    Log("loading answer data.\n");
    vector<int> answerVec;
    loadAnswerData(answerFile, answerVec);
#endif

    Log("begin prediction test.\n");
    logist.predict();

#ifdef TEST
    vector<int> predictVec;
    loadAnswerData(predictFile, predictVec);
    int correctCount = 0;
    for (int j = 0; j < predictVec.size(); j++) {
        if (j < answerVec.size()) {
            if (answerVec[j] == predictVec[j]) {
                correctCount++;
            }
        } else {
            Log("answer size less than the real predicted value.\n");
        }
    }

    double accurate = ((double)correctCount) / answerVec.size();
    Log("the prediction accuracy is ");
    Log(accurate);

#endif

    return 0;
}
