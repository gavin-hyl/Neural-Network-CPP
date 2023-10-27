#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include "Network.h"

using std::vector;

NeuralNetwork::NeuralNetwork(vector<int>& layerSizes) {
    nLayers = layerSizes.size();
    for (int i = 0; i < nLayers - 1; i++) {
        // note that there are only nLayers-1 layers of weights and biases
        weights.push_back(Matrix(layerSizes[i+1], layerSizes[i], RANDOM));
        biases.push_back(Matrix(layerSizes[i+1], 1, RANDOM));
        gradientWeightCost.push_back(Matrix(layerSizes[i+1], layerSizes[i]));
        gradientBiasCost.push_back(Matrix(layerSizes[i+1], 1));
        layerValues.push_back(Matrix(layerSizes[i], 1));
    }
    layerValues.push_back(Matrix(layerSizes[nLayers-1], 1));
}

double NeuralNetwork::cost(Matrix const& output, Matrix& expected) {
    return pow((output-expected).abs(), 2);
}

Matrix NeuralNetwork::gradOutputCost(const Matrix& output, const Matrix& expected) const {
    return (output - expected) * 2;
}

Matrix NeuralNetwork::feedForward(const Matrix& input, bool getMax, bool recordLayerValues) {
    Matrix output = input;
    for (int i = 0; i < nLayers - 1; i++) {
        if (recordLayerValues) {
            layerValues.at(i) = output;
        }
        output = sigma(weights[i] * output + biases[i]);
    }
    output = softmax(output);
    layerValues.at(nLayers-1) = output;
    if (getMax) {
        vector<double> outArray = output.getVectorCol(0);
        int maxIdx = std::max_element(outArray.begin(), outArray.end()) - outArray.begin();
        output.clear();
        output.elements[maxIdx][0] = 1;
    }
    return output;
}

void NeuralNetwork::backPropagate(DataPoint point) {
    Matrix deltaL = gradOutputCost(feedForward(point.data, false, true), point.expected);
    Matrix &dL = deltaL;
    for (int layer = nLayers-2; layer >= 0; layer--) {
        // layer=0 corresponds to the first layer and the weights from the first to second layer
        Matrix &gWC = gradientWeightCost[layer];
        Matrix &gBC = gradientBiasCost[layer];
        gWC = gWC + (dL * layerValues[layer].T());
        gBC = gBC + dL;
        dL = sigmaP(weights[layer].T() * dL);
    }
}

void NeuralNetwork::gradientDescent(vector<DataPoint> dataset, double dBias=DEFAULT_LEARN_STEP, double dWeight=DEFAULT_LEARN_STEP, int epochs=1) {
    double setSize = dataset.size();
    for (int i = 0; i < epochs; i++) {
        for (DataPoint dp : dataset) {
            backPropagate(dp);
        }
        for (int layer = 0; layer < nLayers - 1; layer++) {
            Matrix &w = weights[layer];
            Matrix &b = biases[layer];
            w = w - (gradientWeightCost[layer] * dWeight * (1/setSize));
            // gradientWeightCost[layer].clear();
            b = b - (gradientBiasCost[layer] * dBias * (1/setSize));
            // gradientBiasCost[layer].clear();
        }
        std::cout << "weight[0]\n" << weights[0];
        std::cout << "bias[0]\n" << biases[0];
        std::cout << "epoch " << i+1 << " finished" << std::endl;
        std::cout << "dataset accuracy = " << this->testNetwork(dataset) << "\n-----\n" << std::endl;
    }
}

double NeuralNetwork::testNetwork(vector<DataPoint> dataset) {
    int setSize = dataset.size();
    double nCorrect = 0;
    for (DataPoint dp : dataset) {
        nCorrect += (feedForward(dp.data, true, false) == dp.expected);
    }
    return nCorrect / setSize;
}