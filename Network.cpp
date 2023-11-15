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
            output = weights[i] * output + biases[i];
            // output = ReLU(output);
            output = sigma(output);
            layer_values.at(i) = output;
        }
        output = softmax(output);
        layer_values.back() = output;
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
    return output;
}

void NeuralNetwork::back_propagate(DataPoint point)
{
    Matrix net_out = feed_forward(point.data, false, true);
    Matrix deltaL = gradient_output_cost(net_out, point.expected);
    b_gradients.back() = b_gradients.back() + deltaL;

    for (int layer = num_layers - 2; layer >= 0; layer--)
    {
        // layer=0 corresponds to the first layer and the weights from the first to second layer
        w_gradients.at(layer) = w_gradients[layer] + (deltaL * layer_values.at(layer).T());
        b_gradients.at(layer) = b_gradients[layer] + deltaL;

        // std::cout << w_gradients.at(layer).elements[0][0];
        // (deltaL * layer_values.at(layer).T()).print();
        deltaL = sigma_P(weights.at(layer).T() * deltaL);
        // deltaL = ReLU_P(weights.at(layer).T() * deltaL);
    }
}

void NeuralNetwork::gradient_descent(vector<DataPoint> dataset, double dBias = DEFAULT_LEARN_STEP, double dWeight = DEFAULT_LEARN_STEP, double batchSize = 0)
{
    double setSize = dataset.size();
    batchSize = (batchSize == 0 || batchSize > setSize) ? setSize : batchSize;
    double weightLearnFactor = dWeight / batchSize;
    double biasLeanFactor = dBias / batchSize;
    // std::random_shuffle(dataset.begin(), dataset.end());
    
    for (int batchBegin = 0; batchBegin < setSize; batchBegin += batchSize)
    {
        int miniBatchSize = std::min(batchSize, setSize - batchBegin);
        for (int i = 0; i < miniBatchSize; i++)
        {
            back_propagate(dataset.at(i + batchBegin));
        }
        for (int layer = 0; layer < num_layers - 1; layer++)
        {
            // std::cout << w_gradients.at(layer);
            // std::cout << weights.at(layer).elements[0][0];
            weights.at(layer) = weights.at(layer) - (w_gradients.at(layer) * weightLearnFactor);
            biases.at(layer) = biases.at(layer) - (b_gradients.at(layer) * biasLeanFactor);
            w_gradients.at(layer).clear();
            b_gradients.at(layer).clear();
        }
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