#include "Functions.h"
#include <cmath>
#include <vector>

using std::vector;

Matrix softmax(Matrix v) {
    double sum = 0;
    Matrix result = Matrix(v.rows, 1);
    for (int i = 0; i < v.rows; i++) {
        sum += exp(v.elements[i][0]);
    }
    for (int i = 0; i < v.rows; i++) {
        result.elements[i][0] = exp(v.elements[i][0]) / sum;
    }
}

// calculates d(s(x, k))/d(x, k), assuming that the elements of v have been passed through the softmax functiond
Matrix softmaxP(Matrix v) {
    Matrix result = Matrix(v.rows, v.cols);
    double e;
    for (int i = 0; i < v.rows; i++) {
        for (int j = 0; j < v.cols; j++) {
            e = v.elements[i][j];
            result.elements[i][j] = e * (1-e);
        }
    }
}

double sigma(double d) {
    return 1 / (exp(-d) + 1);
}

Matrix sigma(Matrix m) {
    Matrix result = Matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.elements[i][j] = sigma(m.elements[i][j]);
        }
    }
    return result;
}

// assuming that the elements in m have already been passed throught the sigma function!!!
Matrix sigmaP(Matrix m) {
    Matrix result = Matrix(m.rows, m.cols);
    double e;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            e = m.elements[i][j];
            result.elements[i][j] = e * (1-e);
        }
    }
}

double ReLU(double d) {
    return (d > 0) ? d : 0;
}

Matrix ReLU(Matrix m) {
    Matrix result = Matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.elements[i][j] = ReLU(m.elements[i][j]);
        }
    }
    return result;
}