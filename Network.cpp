#include "Network.h"

using std::vector;

// constructor
NeuralNetwork::NeuralNetwork(vector<int> &topology)
{
    this->topology = topology;
    this->n_layers = topology.size();
    int this_size, next_size;
    for (int i = 1; i < n_layers; i++)
    {   // biases are not added for the input layer but added for the output layer
        biases.push_back(Matrix(topology[i], 1));
        b_grad.push_back(Matrix(topology[i], 1));
        layer_a.push_back(Matrix(topology[i], 1));
        layer_z.push_back(Matrix(topology[i], 1));
    }

    for (int i = 0; i < n_layers - 1; i++)
    {
        weights.push_back(Matrix(topology[i], topology[i+1], RANDOM));
        w_grad.push_back(Matrix(topology[i], topology[i+1]));
    }
}

Matrix NeuralNetwork::output_cost_p(const Matrix &output, const Matrix &expected) const
{
    return (output - expected) * 2;
}

double NeuralNetwork::cost(Matrix const &output, Matrix &expected)
{
    return (output - expected).abs();
}

Matrix NeuralNetwork::feed_forward(const Matrix &input, bool getMax)
{
    Matrix z;
    Matrix a = input;

    int n_transitions = weights.size();     // which is n_layers - 1
    for (int i = 0; i < n_transitions; i++)
    {
        z = weights[i].T() * a + biases[i];
        layer_z.at(i) = z;
        a = (i == n_transitions - 1) ? softmax(z) : sigma(z);
        layer_a.at(i) = a;
    }

    if (getMax)
    {
        Matrix a_cpy = a;
        vector<double> outArray = a_cpy.getVectorCol(0);
        int maxIdx = std::max_element(outArray.begin(), outArray.end()) - outArray.begin();
        a_cpy.clear();
        a_cpy.elements[maxIdx][0] = 1;
        return a_cpy;
    }
    else
    {
        return a;
    }
}

void NeuralNetwork::back_propagate(DataPoint point)
{
    feed_forward(point.data);
    Matrix delta;
    int max_w_idx = n_layers - 2;

    for (int layer = max_w_idx; layer >= 0; layer--)
    {
        if (layer == max_w_idx)
        {
            delta = sigma_P(layer_z.at(layer)).schur(output_cost_p(layer_a.at(layer), point.expected));
            // delta = (layer_a.at(layer) - point.expected).schur(sigma_P(layer_z.at(layer)));
        }
        else
        {
            delta = sigma_P(layer_z.at(layer)).schur(weights.at(layer+1) * delta);
        }
        w_grad.at(layer) = w_grad[layer] + ((layer != 0) ? layer_a[layer-1] * delta.T() : point.data * delta.T());
        b_grad.at(layer) = b_grad[layer] + delta;
    }
}

void NeuralNetwork::update_parameters(double dw, double db)
{
    int n_weights = weights.size();
    for (int layer = 0; layer < n_weights; layer++)
    {
        weights.at(layer) = weights[layer] - (w_grad[layer] * dw);
        biases.at(layer) = biases[layer] - (b_grad[layer] * db);
        w_grad.at(layer).clear();
        b_grad.at(layer).clear();
    } 
}

void NeuralNetwork::gradient_descent(vector<DataPoint> dataset, double dw = DEFAULT_LEARN_STEP, double db = DEFAULT_LEARN_STEP)
{
    for (int i = 0; i < dataset.size(); i++)
    {
        back_propagate(dataset.at(i));
    }
    update_parameters(dw, db);
}

void NeuralNetwork::stochastic_descent(vector<DataPoint> dataset, double dw = DEFAULT_LEARN_STEP, double db = DEFAULT_LEARN_STEP)
{
    for (int i = 0; i < dataset.size(); i++)
    {
        stochastic_descent(dataset.at(i), dw, db);
    }
}

void NeuralNetwork::stochastic_descent(DataPoint point, double dw = DEFAULT_LEARN_STEP, double db = DEFAULT_LEARN_STEP)
{
    back_propagate(point);
    update_parameters(dw, db);
}

void NeuralNetwork::batch_descent(vector<DataPoint> dataset, double dw = DEFAULT_LEARN_STEP, double db = DEFAULT_LEARN_STEP, int batch_size = 0)
{
    double data_size = dataset.size();
    if (batch_size <= 0 || batch_size >= data_size)
    {
        gradient_descent(dataset, dw, db);
        return;
    }
   
    for (int batch_begin = 0; batch_begin < data_size; batch_begin += batch_size)
    {
        int this_batch_size = std::min(double(batch_size), data_size - batch_begin);
        double norm_dw = dw / this_batch_size;
        double norm_db = db / this_batch_size;
        for (int i = 0; i < this_batch_size; i++)
        {
            back_propagate(dataset[i + batch_begin]);
        }
        update_parameters(norm_dw, norm_db);
    }
}


void NeuralNetwork::momentum_descent(vector<DataPoint> dataset, double dw, double db, double gamma)
{
    throw std::logic_error("Not implemented");
}

void NeuralNetwork::nag_descent(vector<DataPoint> dataset, double dw, double db, double gamma)
{
    throw std::logic_error("Not implemented");
}

void NeuralNetwork::adagrad_descent(vector<DataPoint> dataset, double dw, double db, double gamma)
{
    throw std::logic_error("Not implemented");
}

void NeuralNetwork::adadelta_descent(vector<DataPoint> dataset, double dw, double db, double gamma)
{
    throw std::logic_error("Not implemented");
}

void NeuralNetwork::adam_descent(vector<DataPoint> dataset, double dw, double db, double gamma)
{
    throw std::logic_error("Not implemented");
}

double NeuralNetwork::set_accuracy(vector<DataPoint> dataset)
{
    double correct = 0;
    for (DataPoint dp : dataset)
    {
        correct += (feed_forward(dp.data, true) == dp.expected);
    }
    return correct / dataset.size();
}

double NeuralNetwork::set_cost(vector<DataPoint> dataset)
{
    double sum = 0;
    for (DataPoint dp : dataset)
    {
        sum += cost(feed_forward(dp.data), dp.expected);
    }
    return sum;
}