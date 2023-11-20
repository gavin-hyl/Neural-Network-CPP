#include "Network.h"
#include "Read.h"
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <stdio.h>
#include <algorithm>
#include <iomanip>
// #include <stdlib.h>

using std::vector;

Matrix testFunction(vector<double> input, int outDim)
{
    Matrix result = Matrix(outDim, 1);
    double expr = input[0]*input[0] + input[1];
    if (expr > 2)
    {
        result.elements[0][0] = 1;
    }
    // else if (r2 > 5)
    // {
        // result.elements[1][0] = 1;
    // }
    else
    {
        result.elements[1][0] = 1;
    }
    return result;
}

double rand_float(double min, double max)
{
    return (max - min) * rand() / RAND_MAX + min;
}

vector<DataPoint> generateDataSet(int inputDim, int outputDim, int size)
{
    vector<DataPoint> dataset;
    vector<double> input;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < inputDim; j++)
        {
            input.push_back(rand_float(-3, 3));
        }
        DataPoint pt = DataPoint(Matrix::toMatrix(input), testFunction(input, outputDim));
        dataset.push_back(pt);
        input.clear();
    }
    return dataset;
}

void test1(void)
{
    const int IN_DIM = 2;
    const int OUT_DIM = 2;
    // tests to see if the model can classify points inside the unit circle correctly.
    srand((int)time(0));
    int nSamples = 1000;
    vector<DataPoint> trainSet = generateDataSet(IN_DIM, OUT_DIM, nSamples);
    vector<DataPoint> testSet = generateDataSet(IN_DIM, OUT_DIM, nSamples);

    vector<int> layers = {IN_DIM, 10, OUT_DIM};
    NeuralNetwork network = NeuralNetwork(layers);

    double best = 0;
    
    NeuralNetwork best_network = network;
    for (int i = 0; i < 1000; i++)
    {
        DataPoint test = trainSet[i];
        vector<DataPoint> trainset = {test};

        network.gradient_descent(trainset, 1, 1e-1);

        double this_accuracy = network.set_accuracy(trainSet);
        if (this_accuracy > best)
        {
            best = this_accuracy;
            best_network = network;
        }

        if (i % 50 == 0)
        {
            std::cout << "Accuracy at point " << i << " = " << this_accuracy << "\n";
            std::cout << "Best accuracy so far = " << best << "\n";
            std::cout << "===Parameters (W, B)===\n-----\n";
            network.weights[0].print();
            network.biases[0].print();
        }
    }
}

/**
 * @brief RANDOM FUNCTION GEN TESTING
 *
 * @param argc
 * @param argv
 * @return int
 */

int main(int argc, char const *argv[])
{
    test1();
    return 0;
}

// vector<DataPoint> toDataset(vector<vector<double>> inputs, vector<int> outputs, int outputDim) {
//     vector<DataPoint> dataset;
//     int size = outputs.size();
//     for (int i = 0; i < size; i++) {
//         Matrix inputMatrix = Matrix::toMatrix(inputs.at(i));
//         Matrix outputMatrix = Matrix::toBasis(outputs.at(i), outputDim);
//         DataPoint dp = DataPoint(inputMatrix, outputMatrix);
//         dataset.push_back(dp);
//     }
//     return dataset;
// }

// #define IN_DIM 784
// #define OUT_DIM 10

// int main(int argc, char const *argv[])
// {
//     csvFile csv = csvFile("data/mnist_test.csv", "r");
//     vector<vector<double>> mnistInputs;
//     vector<int> mnistOutputs;
//     array<int, 2> dims = csv.getDimensions();
//     int rows = dims[0];
//     int cols = dims[1];
//     for (int i = 0; i < rows; i++) {
//         vector<double> mnistInput;
//         vector<double> row = csv.getDoubleRow(i);
//         for (int j = 1; j < cols; j++) {
//             mnistInput.push_back(row.at(j));
//         }
//         mnistInputs.push_back(mnistInput);
//         mnistOutputs.push_back(row.at(0));
//     }
//     vector<DataPoint> dataset = toDataset(mnistInputs, mnistOutputs, OUT_DIM);

//     vector<int> layers = {IN_DIM, OUT_DIM};
//     NeuralNetwork network = NeuralNetwork(layers);
//     std::cout << std::setprecision (6) << "Initial Accuracy = " << network.set_accuracy(dataset) << std::endl;
//     for (int i = 0; i < 500; i++) {
//         network.batch_descent(dataset, 1E-6, 5E-5, 100);
//         std::cout << "train set accuracy = " << network.set_accuracy(dataset) << std::endl;
//         printf("Weights:\n");
//         network.weights[0].print();
//         printf("Biases\n");
//         network.biases[0].print();
//     }

//     return 0;
// }
