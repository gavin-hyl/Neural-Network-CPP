#pragma once

#include <vector>
#include <iostream>
#include "Functions.h"
#include "Data.h"

#define DEFAULT_LEARN_STEP 0.001

using std::vector;

class NeuralNetwork {

    public:
        vector<Matrix> weights;             // all weights between layers, n-1 in total
        vector<Matrix> biases;              // all biases between layers, n-1 in total
        vector<Matrix> gradientWeightCost;  // intermediate variable, used to pass the results from back propagation to gradient descent
        vector<Matrix> gradientBiasCost;    // intermediate variable, used to pass the results from back propagation to gradient descent
        vector<Matrix> layerValues;         // includes values of all layers, including the input and output, n in total
        int nLayers;                        // total number of layers

        /**
         * @brief 
         * 
         * @param output 
         * @param expected 
         * @return double 
         */
        double cost(Matrix const& output, Matrix& expected);
        /**
         * @brief 
         * 
         * @param output 
         * @param expected 
         * @return Matrix 
         */
        Matrix gradOutputCost(const Matrix& output, const Matrix& expected) const;

    public:
        /**
         * @brief Construct a new Neural Network object
         * 
         * @param layerSizes sizes of all the layers, including input and output
         */
        NeuralNetwork(vector<int>& layerSizes);

        /**
         * @brief Calculate the output of the network given a single input
         * 
         * @param input input data
         * @param getMax if true, returns a matrix with the highest probability index set to 1 and all others set to zero
         * @param recordLayerValues if true, records the layer values internally
         * @return (Matrix) the output
         */
        Matrix feedForward(const Matrix& input, bool getMax=false, bool recordLayerValues=false);
        
        /**
         * @brief Calculate the gradient of the cost function at one data point, stores result in gradient Weight/Bias Cost
         * 
         * @param point the data point
         */
        void backPropagate(DataPoint point);

        /**
         * @brief Vanilla gradient descent algorithm, using the gradient calculated by backPropagate
         * 
         * @param learnStep step size
         */
        void gradientDescent(vector<DataPoint> dataset, double, double, int);

        /**
         * @brief Test network accuracy based on a testing set
         * 
         * @param dataset the testing set
         * @return (double) network accuracy
         */
        double testNetwork(vector<DataPoint> dataset);
};