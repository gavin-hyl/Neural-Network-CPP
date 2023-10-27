#pragma once
#include "Matrix.h"

Matrix softmax(const Matrix& v);
double sigma(double d);
Matrix sigma(const Matrix& m);
Matrix sigmaP(Matrix m);
double ReLU(double d);
Matrix ReLU(Matrix v);
