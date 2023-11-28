#include "Network.h"
#include "Read.h"
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
*/

VectorXd classification(VectorXd input)
{
    VectorXd label = VectorXd::Zero(2);
    label((input.norm() > 0.5)) = 1;
    return label;
}

vector<DataPoint> generate_set(const int samples, const int in_dim)
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

void record_values(std::string path, const vector<double> &data)
{
    std::ofstream file(path);
    file << "header\n";
    for (double d : data)
    {
        file << d << "\n";
    }
    file.close();
}

int main()
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
        nn.batch_descent(train, 1, 0, 30);
        nn.evaluate(train);
        accuracies.push_back(nn.set_accuracy(test));
        costs.push_back(nn.set_cost(test));
    }
    record_values("accuracy", accuracies);
    record_values("cost", costs);
    return 0;
}
