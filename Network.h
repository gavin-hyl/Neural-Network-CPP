#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include "Eigen/Dense"
#include "Functions.h"
#include "Data.h"

#define DEFAULT_LEARN_STEP 1e-3

using std::vector;
using Eigen::VectorXd;
using Eigen::MatrixXd;

class NeuralNetwork
{
public:
    vector<int> topology;
    vector<MatrixXd> weights;              // all weights between layers, n-1 in total
    vector<MatrixXd> biases;               // all biases between layers, n in total
    vector<MatrixXd> w_grad;
    vector<MatrixXd> b_grad;
    vector<VectorXd> layer_a;
    vector<VectorXd> layer_z;
    int n_layers;                      // total number of layers

    /**
     * @brief
     *
     * @param output
     * @param expected
     * @return double
     */
    double cost(const VectorXd &output, const VectorXd &expected) const;

    double cost(const vector<DataPoint> &dataset) const;

    VectorXd output_cost_p(const VectorXd &output, const VectorXd &expected) const;

    /**
     * @brief Construct a new Neural Network object
     *
     * @param layerSizes sizes of all the layers, including input and output
     */
    NeuralNetwork(const vector<int> &layerSizes);

    /**
     * @brief Calculate the output of the network given a single input
     *
     * @param input input data
     * @param get_max if true, returns a matrix with the highest probability index set to 1 and all others set to zero
     * @param record_layer_values if true, records the layer values internally
     * @return (Matrix) the output
     */
    VectorXd feed_forward(const VectorXd &input, const bool get_max = false);

    /**
     * @brief Calculate the gradient of the cost function at one data point, stores result in gradient Weight/Bias Cost
     *
     * @param point the data point
     */
    void back_propagate(const DataPoint &point);

    void update_parameters(const double d_weight, const double d_bias);

    // DESCENT ALGORITHMS AND OPTIMIZERS
    void gradient_descent(const vector<DataPoint> &dataset, double, double);
    void batch_descent(const vector<DataPoint> &dataset, double, double, int);
    void stochastic_descent(const vector<DataPoint> &dataset, double, double);
    void stochastic_descent(const DataPoint &point, double, double);
    void momentum_descent(const vector<DataPoint> &dataset, double, double, double);
    void nag_descent(const vector<DataPoint> &dataset, double, double, double);
    void adagrad_descent(const vector<DataPoint> &dataset, double, double);
    void adadelta_descent(const vector<DataPoint> &dataset, double, double, double);
    void adam_descent(const vector<DataPoint> &dataset, double, double, double, double);

    double set_accuracy(const vector<DataPoint> &dataset);
    double set_cost(const vector<DataPoint> &dataset);

    void evaluate(const vector<DataPoint> &dataset);
};