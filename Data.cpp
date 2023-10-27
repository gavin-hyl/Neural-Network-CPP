#include "Matrix.h"
#include "Data.h"
#include <cmath>
#include <vector>

using std::vector;

DataPoint::DataPoint(const Matrix data) {
    this->data = data;
    this->expected = Matrix(1, 1);
    hasExpected = false;
}

DataPoint::DataPoint(Matrix input, Matrix expected) {
    this->data = input;
    this->expected = expected;
    hasExpected = true;
}

// DataSet::DataSet(const vector<DataPoint>& points) {
//     this->points = points;
// }

// DataSet::DataSet(const vector<Matrix>& inputs, const vector<Matrix>& expected) {
    
// }