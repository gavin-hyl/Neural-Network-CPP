#include "Functions.h"
#include <cmath>
#include <vector>

using std::vector;

Matrix softmax(const Matrix& v) {
    double sum = 0;
    Matrix result = Matrix(v.rows, 1);
    for (int i = 0; i < v.rows; i++) {
        result.elements[i][0] = exp(v.elements[i][0]);
        sum += result.elements[i][0];
    }
    for (int i = 0; i < v.rows; i++) {
        result.elements[i][0] = result.elements[i][0] / sum;
    }
    return result;
}

// calculates d(s(x, k))/d(x, k), assuming that the elements of v have been passed through the softmax function
Matrix softmax_P(Matrix v) {
    Matrix result = Matrix(v.rows, v.cols);
    double e;
    for (int i = 0; i < v.rows; i++) {
        for (int j = 0; j < v.cols; j++) {
            e = v.elements[i][j];
            result.elements[i][j] = e * (1-e);
        }
    }
    return result;
}

double sigma(double d) {
    return 1 / (exp(-d) + 1);
}

Matrix sigma(const Matrix& m) {
    Matrix result = Matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.elements[i][j] = sigma(m.elements[i][j]);
        }
    }
    return result;
}

// assuming that the elements in m have already been passed through the sigma function!!!
Matrix sigma_P(Matrix m) {
    Matrix result = Matrix(m.rows, m.cols);
    double e;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            e = m.elements[i][j];
            result.elements[i][j] = e * (1-e);
        }
    }
    return result;
}

inline double ReLU(double d) {
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

Matrix ReLU_P(Matrix m) {
    Matrix result = Matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.elements[i][j] = m.elements[i][j] > 0;
        }
    }
    return result;
}