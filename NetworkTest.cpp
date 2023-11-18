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
    if (input[0] + input[1] > 0)
    {
        result.elements[0][0] = 1;
        // } else if (expr > 0) {
        //     result.elements[1][0] = 1;
        // } else if (expr > -2) {
        //     result.elements[2][0] = 1;
    }
    else
    {
        result.elements[outDim - 1][0] = 1;
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
            input.push_back(rand_float(-10, 10));
        }
        DataPoint pt = DataPoint(Matrix::toMatrix(input), testFunction(input, outputDim));
        dataset.push_back(pt);
        input.clear();
    }
    return dataset;
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
    const int IN_DIM = 2;
    const int OUT_DIM = 2;
    // tests to see if the model can classify points inside the unit circle correctly.
    srand((int)time(0));
    int nSamples = 10000;
    vector<DataPoint> trainSet = generateDataSet(IN_DIM, OUT_DIM, nSamples);
    vector<DataPoint> testSet = generateDataSet(IN_DIM, OUT_DIM, nSamples);

    vector<int> layers = {IN_DIM, OUT_DIM};
    NeuralNetwork network = NeuralNetwork(layers);

    for (int i = 0; i < 20; i++)
    {
        std::cout << "---------------------------------------------\n";
    // std::cout << "Data Point\n";
    // trainSet[i].print();
    // vector<DataPoint> trainset = {trainSet[i]};

    // std::cout << "Initial Prediction\n";
    // network.feed_forward(trainSet[0].data, false, true).print();

    // std::cout << "Parameters\n";
    // network.weights[0].print();
    // network.biases[0].print();

    network.back_propagate(trainSet[i]);

    // std::cout << "Gradients\n";
    // network.w_gradients[0].print();
    // network.b_gradients[0].print();

    network.update_parameters(1E-2, 1E-2);
    // std::cout << "Updated parameters\n";
    // network.weights[0].print();
    // network.biases[0].print();

    std::cout << "After Descent, test accuracy =" << network.test_network(testSet) << "\n";
    // _sleep(1000);
    }

    // std::cout << std::setprecision(4) << "Initial Accuracy (train)= " << network.test_network(trainSet) << std::endl;
    // for (int i = 0; i < 100; i++)
    // {
    //     network.batch_descent(trainSet, 0, 5E-5, 1);
    //     std::cout << "train set accuracy = " << network.test_network(trainSet) << std::endl;
    //     std::cout << "train set cost = " << network.cost(trainSet) << std::endl;
    //     if (i % 10 == 0)
    //     {
    //         printf("Weights:\n");
    //         network.weights[0].print();
    //         printf("Biases\n");
    //         network.biases[0].print();
    //     }
    //     // _sleep(1000);
    // }
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
//     std::cout << std::setprecision (6) << "Initial Accuracy = " << network.test_network(dataset) << std::endl;
//     for (int i = 0; i < 500; i++) {
//         network.batch_descent(dataset, 1E-6, 5E-5, 100);
//         std::cout << "train set accuracy = " << network.test_network(dataset) << std::endl;
//         printf("Weights:\n");
//         network.weights[0].print();
//         printf("Biases\n");
//         network.biases[0].print();
//     }

//     return 0;
// }
