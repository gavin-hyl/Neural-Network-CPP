#include "Tests.h"
#include "Network.h"
#include "Read.h"
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;
using std::vector;

/*
    MNIST handwritten digits dataset.
*/

static vector<DataPoint> set_from_csv(std::string path)
{
    CSV csv_file = CSV(path);
    vector<DataPoint> dataset;
    for (int i = 0; i < csv_file.getDimensions()[0]; i++)
    {
        vector<double> row = csv_file.getDoubleRow(i);
        int num_label = row[0];
        row.erase(row.begin());
        VectorXd label = VectorXd::Zero(10);
        label(num_label) = 1;
        VectorXd input = Eigen::Map<Eigen::Matrix<double, 784, 1>>(row.data());
        dataset.push_back({input, label});
    }
    return dataset;
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

void test_mnist()
{
    std::cout << std::setprecision(5);
    srand(time(0));
    int in_dim = 784;
    NeuralNetwork nn = NeuralNetwork({in_dim, 10});
    vector<DataPoint> test = set_from_csv("Data/mnist_train.csv");
    std::cout << "set length = " << test.size() << "\n";
    vector<double> accuracies;
    vector<double> costs;

    // std::cout << test[0].input << "\n";
    // std::cout << test[0].label << "\n";

    for (int i = 0; i < 10; i++)
    {
        nn.evaluate(test);
        std::cout << nn.weights[0].norm() << "\n";
        nn.momentum_descent(test, 0.05, 0.01, 0.9);
        accuracies.push_back(nn.set_accuracy(test));
        costs.push_back(nn.set_cost(test));
    }
    record_values("Visualization/accuracy", accuracies);
    record_values("Visualization/cost", costs);
}
