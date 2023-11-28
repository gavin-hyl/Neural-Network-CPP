#include "Functions.h"
#include <cmath>
#include <vector>

using std::vector;

MatrixXd broadcast(const MatrixXd &M, double (*f)(double))
{
    int row = M.rows();
    int col = M.cols();
    MatrixXd result(row, col);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            result(i, j) = f(M(i, j));
        }
    }
    return result;
}

void broadcast_inplace(MatrixXd &M, double (*f)(double))
{
    int row = M.rows();
    int col = M.cols();
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            M(i, j) = f(M(i, j));
        }
    }
}

MatrixXd softmax(const MatrixXd &M)
{
    MatrixXd result(M.rows(), M.cols());
    broadcast(result, exp);
    double sum = result.sum();
    result /= sum;
    return result;
}

MatrixXd softmax_P(const MatrixXd &M)
{
    MatrixXd result = softmax(M);
    broadcast_inplace(result, [] (double e) {return e * (1-e);});
    return result;
}

double sigma(double d)
{
    return 1 / (exp(-d) + 1);
}

double sigma_p(const double d)
{
    double tmp = sigma(d);
    return tmp * (1-tmp);
}

double ReLU(double d)
{
    return (d > 0) ? d : 0;
}

double ReLU_p(const double d)
{
    return (d > 0);
}