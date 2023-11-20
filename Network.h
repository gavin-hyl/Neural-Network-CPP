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
    vector<int> topology;
    vector<Matrix> weights;              // all weights between layers, n-1 in total
    vector<Matrix> biases;               // all biases between layers, n in total
    vector<Matrix> w_grad;
    vector<Matrix> b_grad;
    vector<Matrix> layer_a;
    vector<Matrix> layer_z;
    int n_layers;                      // total number of layers

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
    Matrix output_cost_p(const Matrix &output, const Matrix &expected) const;

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
    Matrix feed_forward(const Matrix &input, bool get_max = false);

    /**
     * @brief Calculate the gradient of the cost function at one data point, stores result in gradient Weight/Bias Cost
     *
     * @param point the data point
     */
    void back_propagate(DataPoint point);

    void update_parameters(double d_weight, double d_bias);

    // DESCENT ALGORITHMS AND OPTIMIZERS
    void gradient_descent(vector<DataPoint> dataset, double, double);
    void batch_descent(vector<DataPoint> dataset, double, double, int);
    void stochastic_descent(vector<DataPoint> dataset, double, double);
    void stochastic_descent(DataPoint point, double, double);
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
    double set_accuracy(vector<DataPoint> dataset);
    double set_cost(vector<DataPoint> dataset);
};