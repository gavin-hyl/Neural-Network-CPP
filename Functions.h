#pragma once
#include "Eigen/Dense"

using Eigen::MatrixXd;

MatrixXd broadcast(const MatrixXd &M, double (*f)(double));
void broadcast_inplace(MatrixXd &M, double (*f)(double));
MatrixXd softmax(const MatrixXd &v);
MatrixXd softmax_p(const MatrixXd &v);
double sigma(const double d);
double sigma_p(const double d);
double ReLU(const double d);
double ReLU_p(const double d);
