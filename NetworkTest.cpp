#include "Network.h"
#include "Read.h"
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <stdio.h>
#include <algorithm>
#include <iomanip>

using std::vector;
using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::MatrixXd;

/*
Point Classification, inspiration from Sebastian League's video on Neural Networks.
*/

VectorXd classification(VectorXd input)
{
    Vector2d label = Vector2d::Zero();
    if (input.sum() > 0)
    {
        label(0) = 1;
    }
    else
    {
        label(1) = 1;
    }
    return label;
}

vector<DataPoint> generate_set(const int samples, const int in_dim)
{
    vector<DataPoint> set;
    for (int i = 0; i < samples; i++)
    {
        VectorXd input = VectorXd::Random(in_dim);  // from -1 to 1
        DataPoint dp = {input, classification(input)};
        set.push_back(dp);
    }
    return set;
}

int main()
{
    srand(time(0));
    int in_dim = 2;
    NeuralNetwork nn = NeuralNetwork({in_dim, 2});
    vector<DataPoint> set = generate_set(1000, in_dim);
    nn.evaluate(set);
    nn.batch_descent(set, 1, 0.1, 10);
    nn.evaluate(set);
    return 0;
}
