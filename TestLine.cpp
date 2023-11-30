#include "Tests.h"
#include "Network.h"
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;
using std::vector;

/*
Easiest test possible.
*/

static VectorXd classification(VectorXd input)
{
    VectorXd label = VectorXd::Zero(3);
    if (input.sum() > 1)
    {
        label(0) = 1;
    }
    else if (input.sum() < -1)
    {
        label(1) = 1;
    }
    else
    {
        label(2) = 1;
    }
    return label;
}

static vector<DataPoint> generate_set(const int samples, const int in_dim)
{
    vector<DataPoint> set;
    for (int i = 0; i < samples; i++)
    {
        VectorXd input = VectorXd::Random(in_dim); // from -1 to 1
        DataPoint dp = {input, classification(input)};
        set.push_back(dp);
    }
    return set;
}

static void record_values(std::string path, const vector<double> &data)
{
    std::ofstream file(path);
    file << "header\n";
    for (double d : data)
    {
        file << d << "\n";
    }
    file.close();
}

void test_line()
{
    srand(time(0));
    int in_dim = 2;
    int out_dim = 3;
    NeuralNetwork nn = NeuralNetwork({in_dim, 3, out_dim}); // the hidden layer was crucial - without it, the model could not improve past 80% accuracy.
    vector<DataPoint> train = generate_set(1000, in_dim);
    vector<DataPoint> test = generate_set(100, in_dim);
    vector<double> accuracies;
    vector<double> costs;

    nn.evaluate(test);
    for (int i = 0; i < 500; i++)
    {
        nn.momentum_descent(train, 0.1, 0.01, 0.5);
        nn.evaluate(train);
        accuracies.push_back(nn.set_accuracy(test));
        costs.push_back(nn.set_cost(test));
    }
    record_values("Visualization/accuracy", accuracies);
    record_values("Visualization/cost", costs);
}
