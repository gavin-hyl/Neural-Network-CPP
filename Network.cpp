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

Matrix NeuralNetwork::feedForward(Matrix input, bool getMax, bool recordLayerValues) {
    for (int i = 0; i < nLayers - 1; i++) {
        if (recordLayerValues) {
            layerValues.at(i) = input;
        }
        input = sigma(weights[i] * input + biases[i]);
    }
    Matrix output = softmax(input);
    layerValues.at(nLayers-1) = output;
    if (getMax) {
        vector<double> outArray = output.getVectorCol(0);
        int maxIdx = std::max_element(outArray.begin(), outArray.end()) - outArray.begin();
        output = Matrix(outArray.size(), 1);
        output.elements[maxIdx][0] = 1; // all other elements are 0
        return output;
    }
    return output;
}

void NeuralNetwork::backPropagate(DataPoint point) {
    Matrix deltaL = gradOutputCost(feedForward(point.data, false, true), point.expected);
    Matrix &dL = deltaL;
    for (int layer = nLayers-2; layer >= 0; layer--) {
        // layer=0 corresponds to the first layer and the weights from the first to second layer
        // std::cout << deltaL;
        Matrix &gWC = gradientWeightCost[layer];
        Matrix &gBC = gradientBiasCost[layer];
        gWC = (dL * layerValues[layer].T());
        gBC = dL;
        dL = sigmaP(weights[layer].T() * dL);
    }
}

void NeuralNetwork::gradientDescent(vector<DataPoint> dataset, double learnStep=DEFAULT_LEARN_STEP, int epochs=1) {
    int setSize = dataset.size();
    for (int i = 0; i < epochs; i++) {
        for (DataPoint dp : dataset) {
            backPropagate(dp);
            // std::cout << gradientWeightCost[0];        
            // std::cout << weights[0];
            for (int layer = 0; layer < nLayers - 1; layer++) {
                Matrix &w = weights[layer];
                Matrix &b = biases[layer];
                w = w - (gradientWeightCost[layer] * learnStep);
                // gradientWeightCost[layer].clear();
                b = b - (gradientBiasCost[layer] * learnStep);
                // gradientBiasCost[layer].clear();
            }
        }
        std::cout << "epoch " << i+1 << " finished" << std::endl;
        std::cout << "test accuracy = " << this->testNetwork(dataset) << std::endl;
    }
}

double NeuralNetwork::testNetwork(vector<DataPoint> dataset) {
    int setSize = dataset.size();
    double nCorrect = 0;
    for (DataPoint dp : dataset) {
        nCorrect += (feedForward(dp.data, true, false) == dp.expected);
    }
    nCorrect = nCorrect / setSize;

    return nCorrect;
}