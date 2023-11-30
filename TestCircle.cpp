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
Point Classification, inspired by Sebastian League's video on Neural Networks.
This test demonstrates the ability of the mini-batch algorithm to jump out of local minima.
*/

static VectorXd classification(VectorXd input)
{
    VectorXd label = VectorXd::Zero(2);
    label((input.norm() > 0.5)) = 1;
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

void test_circle()
{
    srand(time(0));
    int in_dim = 2;
    NeuralNetwork nn = NeuralNetwork({in_dim, 6, 2});
    vector<DataPoint> train = generate_set(3000, in_dim);
    vector<DataPoint> test = generate_set(500, in_dim);
    vector<double> accuracies;
    vector<double> costs;

    nn.evaluate(test);
    for (int i = 0; i < 300; i++)
    {
        nn.momentum_descent(train, 0.1, 0.01, 0.5);
        nn.evaluate(train);
        accuracies.push_back(nn.set_accuracy(test));
        costs.push_back(nn.set_cost(test));
    }
    record_values("Visualization/accuracy", accuracies);
    record_values("Visualization/cost", costs);
}
