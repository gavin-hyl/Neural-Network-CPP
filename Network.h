#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include "Functions.h"
#include "Data.h"

#define DEFAULT_LEARN_STEP 1e-3

using std::vector;

class NeuralNetwork
{
private:
    Matrix inner_activation(Matrix M);
    Matrix inner_activation_P(Matrix M);
    Matrix output_activation(Matrix M);
    Matrix output_activation_P(Matrix M);

public:
    vector<Matrix> weights;              // all weights between layers, n-1 in total
    vector<Matrix> biases;               // all biases between layers, n in total
    vector<Matrix> w_gradients; // intermediate variable, used to pass the results from back propagation to gradient descent
    vector<Matrix> b_gradients;   // intermediate variable, used to pass the results from back propagation to gradient descent
    vector<Matrix> layer_values;         // includes values of all layers, including the input and output, n in total
    int num_layers;                      // total number of layers
    char inner_activation_type;
    char output_activation_type;

    /**
     * @brief
     *
     * @param output
     * @param expected
     * @return double
     */
    double cost(Matrix const &output, Matrix &expected);

    double cost(vector<DataPoint> dataset);

    /**
     * @brief
     *
     * @param output
     * @param expected
     * @return Matrix
     */
    Matrix gradient_output_cost(const Matrix &output, const Matrix &expected) const;

    /**
     * @brief Construct a new Neural Network object
     *
     * @param layerSizes sizes of all the layers, including input and output
     */
    NeuralNetwork(vector<int> &layerSizes);

    /**
     * @brief Calculate the output of the network given a single input
     *
     * @param input input data
     * @param get_max if true, returns a matrix with the highest probability index set to 1 and all others set to zero
     * @param record_layer_values if true, records the layer values internally
     * @return (Matrix) the output
     */
    Matrix feed_forward(const Matrix &input, bool get_max = false, bool record_layer_values = false);

    /**
     * @brief Calculate the gradient of the cost function at one data point, stores result in gradient Weight/Bias Cost
     *
     * @param point the data point
     */
    void back_propagate(DataPoint point);

    void back_propagate(vector<DataPoint> dataset);

    /**
     * @brief Vanilla gradient descent algorithm, using the gradient calculated by back_propagate
     *
     * @param learnStep step size
     */
    void gradient_descent(vector<DataPoint> dataset, double, double, double);

    /**
     * @brief 
     * 
     * @param dataset 
     * @param db bias learning rate
     * @param dw weight learning rate
     * @param batch_size 
     * @param gamma momentum learning hyperparameter
     */
    void momentum_descent(vector<DataPoint> dataset, double db, double dw, double gamma);
    void nag_descent(vector<DataPoint> dataset, double db, double dw, double gamma);
    void adagrad_descent(vector<DataPoint> dataset, double db, double dw, double gamma);
    void adadelta_descent(vector<DataPoint> dataset, double db, double dw, double gamma);

    
    void adam_descent(vector<DataPoint> dataset, double db, double dw, double gamma);

    /**
     * @brief Test network accuracy based on a testing set
     *
     * @param dataset the testing set
     * @return (double) network accuracy
     */
    double test_network(vector<DataPoint> dataset);
};