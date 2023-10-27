#pragma once
#include "Matrix.h"
#include <cmath>
#include <vector>

using std::vector;

class DataPoint {
    public:
        Matrix data;
        Matrix expected;
        bool hasExpected;
        DataPoint(const Matrix data);
        DataPoint(Matrix input, Matrix expected);
};

// class DataSet {
//     public:
//         vector<DataPoint> points;
//         DataSet(const vector<DataPoint>& points);
//         DataSet(const vector<Matrix>& inputs, const vector<Matrix>& expected);
//         vector<DataPoint> next(int n, bool wrap=false);

// };