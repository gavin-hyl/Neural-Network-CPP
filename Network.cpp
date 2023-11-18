#include "Network.h"

using std::vector;

// constructor
NeuralNetwork::NeuralNetwork(vector<int> &topology)
{
    num_layers = topology.size();
    int this_size, next_size;
    for (int i = 0; i < num_layers - 1; i++)
    {
        // note that there are only num_layers-1 layers of weights
        this_size = topology[i];
        next_size = topology[i + 1];

        weights.push_back(Matrix(next_size, this_size, RANDOM));
        biases.push_back(Matrix(next_size, 1));

        w_gradients.push_back(Matrix(next_size, this_size));
        b_gradients.push_back(Matrix(next_size, 1));

        layer_values.push_back(Matrix(this_size, 1));
    }

    layer_values.push_back(Matrix(next_size, 1));
}

Matrix NeuralNetwork::gradient_output_cost(const Matrix &output, const Matrix &expected) const
{
    return (output - expected) * 2;
}

double NeuralNetwork::cost(Matrix const &output, Matrix &expected)
{
    return (output - expected).abs();
}

double NeuralNetwork::cost(vector<DataPoint> dataset)
{
    double sum = 0;
    for (DataPoint dp : dataset)
    {
        sum += cost(feed_forward(dp.data), dp.expected);
    }
    return sum;
}

Matrix NeuralNetwork::feed_forward(const Matrix &input, bool getMax, bool recordLayerValues)
{
    Matrix output = input;
    if (recordLayerValues)
    {
        for (int i = 0; i < num_layers - 1; i++)
        {
            layer_values.at(i) = output;
            output = weights[i] * output + biases[i];
            // output = ReLU(output);
            output = sigma(output);
        }
        layer_values.back() = output;
        output = softmax(output);
    }
    else
    {
        for (int i = 0; i < num_layers - 1; i++)
        {
            output = weights[i] * output + biases[i];
            // output = ReLU(output);
            output = sigma(output);
        }
        output = softmax(output);
    }

    if (getMax)
    {
        vector<double> outArray = output.getVectorCol(0);
        int maxIdx = std::max_element(outArray.begin(), outArray.end()) - outArray.begin();
        output.clear();
        output.elements[maxIdx][0] = 1;
    }
        // std::cout << "Layer value [0]\n";
        // layer_values.at(0).print();
    return output;
}

void NeuralNetwork::back_propagate(DataPoint point)
{
    Matrix net_out = feed_forward(point.data, false, true);
    Matrix deltaL = softmax_P(gradient_output_cost(net_out, point.expected));
    // std::cout << "Delta L\n";
    // deltaL.print();
    // b_gradients.back() = b_gradients.back() + deltaL;

    for (int layer = num_layers - 2; layer >= 0; layer--)
    {
        // layer=0 corresponds to the first layer and the weights from the first to second layer
        // std::cout << "layer values [0]\n";
        // layer_values.at(layer).print();
        w_gradients.at(layer) = (deltaL * layer_values.at(layer).T());
        b_gradients.at(layer) = deltaL;

        // std::cout << w_gradients.at(layer).elements[0][0];
        // (deltaL * layer_values.at(layer).T()).print();
        deltaL = sigma_P(weights.at(layer).T() * deltaL);
    }
}

void NeuralNetwork::update_parameters(double d_weight, double d_bias)
{
    for (int layer = 0; layer < num_layers - 1; layer++)
    {
        weights.at(layer) = weights[layer] - (w_gradients[layer] * d_weight);
        biases.at(layer) = biases[layer] - (b_gradients[layer] * d_bias);
        w_gradients.at(layer).clear();
        b_gradients.at(layer).clear();
    }
}

void NeuralNetwork::stochastic_descent(vector<DataPoint> dataset, double dBias = DEFAULT_LEARN_STEP, double dWeight = DEFAULT_LEARN_STEP)
{
    for (int i = 0; i < dataset.size(); i++)
    {
        stochastic_descent(dataset.at(i), dWeight, dBias);
    }
}

void NeuralNetwork::stochastic_descent(DataPoint point, double dBias = DEFAULT_LEARN_STEP, double dWeight = DEFAULT_LEARN_STEP)
{
    back_propagate(point);
    update_parameters(dWeight, dBias);
}

void NeuralNetwork::batch_descent(vector<DataPoint> dataset, double dBias = DEFAULT_LEARN_STEP, double dWeight = DEFAULT_LEARN_STEP, double batch_size = 0)
{
    double data_size = dataset.size();
    batch_size = (batch_size == 0 || batch_size > data_size) ? data_size : batch_size;
    double norm_d_weight = dWeight / batch_size;
    double norm_d_bias = dBias / batch_size;
    // std::random_shuffle(dataset.begin(), dataset.end());
    
    for (int batch_begin = 0; batch_begin < data_size; batch_begin += batch_size)
    {
        int this_batch_size = std::min(batch_size, data_size - batch_begin);
        for (int i = 0; i < this_batch_size; i++)
        {
            back_propagate(dataset.at(i + batch_begin));
        }
        update_parameters(norm_d_weight, norm_d_bias);
    }
}


void NeuralNetwork::momentum_descent(vector<DataPoint> dataset, double db, double dw, double gamma)
{
    double set_size = dataset.size();

    vector<Matrix> w_previous (num_layers - 1);
    vector<Matrix> b_previous (num_layers - 1);

    for (int i = 0; i < set_size; i++)
    {
        back_propagate(dataset[i]);
        for (int layer = 0; layer < num_layers - 1; layer++)
        {
            if (i != 0)
            {
                weights.at(layer) = weights.at(layer) - (w_gradients.at(layer) * dw) - w_previous.at(layer) * gamma;
                biases.at(layer) = biases.at(layer) - (b_gradients.at(layer) * db) - b_previous.at(layer) * gamma;
            }
            else
            {
                weights.at(layer) = weights.at(layer) - (w_gradients.at(layer) * dw);
                biases.at(layer) = biases.at(layer) - (b_gradients.at(layer) * db);
            }
            w_previous.at(layer) = w_gradients[layer];
            b_previous.at(layer) = b_gradients[layer];
            w_gradients.at(layer).clear();
            b_gradients.at(layer).clear();
        }
    }
}

void NeuralNetwork::nag_descent(vector<DataPoint> dataset, double db, double dw, double gamma)
{

}

void NeuralNetwork::adagrad_descent(vector<DataPoint> dataset, double db, double dw, double gamma)
{
    
}

void NeuralNetwork::adadelta_descent(vector<DataPoint> dataset, double db, double dw, double gamma)
{
    
}

void NeuralNetwork::adam_descent(vector<DataPoint> dataset, double db, double dw, double gamma)
{
    
}

double NeuralNetwork::test_network(vector<DataPoint> dataset)
{
    double nCorrect = 0;
    for (DataPoint dp : dataset)
    {
        nCorrect += (feed_forward(dp.data, true, false) == dp.expected);
    }
    return nCorrect / dataset.size();
}