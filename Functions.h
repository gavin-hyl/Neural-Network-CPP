#pragma once
#include "Matrix.h"

Matrix softmax(const Matrix& v);
Matrix softmax_P(Matrix v);
double sigma(double d);
Matrix sigma(const Matrix& m);
Matrix sigma_P(const Matrix& m);
double ReLU(double d);
Matrix ReLU(Matrix v);
Matrix ReLU_P(Matrix v);
