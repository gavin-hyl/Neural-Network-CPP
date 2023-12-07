#include "Network.h"

// constructor
NeuralNetwork::NeuralNetwork(const vector<int> &topology)
{
    this->topology = topology;
    this->n_layers = topology.size();
    int this_size, next_size;
    for (int i = 1; i < n_layers; i++)
    { // biases are not added for the input layer but added for the output layer
        biases.push_back(VectorXd::Random(topology[i]));
        b_grad.push_back(VectorXd::Zero(topology[i]));
        layer_a.push_back(VectorXd::Zero(topology[i]));
        layer_z.push_back(VectorXd::Zero(topology[i]));
    }

    for (int i = 0; i < n_layers - 1; i++)
    {
        weights.push_back(MatrixXd::Random(topology[i], topology[i + 1]));
        w_grad.push_back(MatrixXd::Zero(topology[i], topology[i + 1]));
    }
}

VectorXd NeuralNetwork::output_cost_p(const VectorXd &output, const VectorXd &expected) const
{
    return (output - expected) * 2;
}

double NeuralNetwork::cost(const VectorXd &output, const VectorXd &expected) const
{
    return (output - expected).squaredNorm();
}

VectorXd NeuralNetwork::feed_forward(const VectorXd &input, const bool getMax)
{
    VectorXd z;
    VectorXd a = input;

    int n_transitions = weights.size(); // which is n_layers - 1
    for (int i = 0; i < n_transitions; i++)
    {
        z = weights[i].transpose() * a + biases[i];
        layer_z.at(i) = z;
        a = broadcast(z, sigma);
        layer_a.at(i) = a;
    }

    if (getMax)
    {
        Eigen::Index max_row, max_col;
        a.maxCoeff(&max_row, &max_col);
        a.setZero();
        a(max_row) = 1;
    }
    return a;
}

void NeuralNetwork::back_propagate(const DataPoint &point)
{
    feed_forward(point.input);
    VectorXd delta;
    int max_w_idx = n_layers - 2;

    for (int layer = max_w_idx; layer >= 0; layer--)
    {
        if (layer == max_w_idx)
        {
            delta = broadcast(layer_z[layer], sigma_p).cwiseProduct(output_cost_p(layer_a.at(layer), point.label));
        }
        else
        {
            delta = broadcast(layer_z[layer], sigma_p).cwiseProduct(weights[layer + 1] * delta);
        }
        w_grad.at(layer) = w_grad[layer] + ((layer != 0) ? layer_a[layer - 1] * delta.transpose() : point.input * delta.transpose());
        b_grad.at(layer) = b_grad[layer] + delta;
    }
}

void NeuralNetwork::update_parameters(double dw, double db)
{
    int n_weights = weights.size();
    for (int layer = 0; layer < n_weights; layer++)
    {
        weights.at(layer) -= w_grad[layer] * dw;
        biases.at(layer) -= b_grad[layer] * db;
        w_grad.at(layer).setZero();
        b_grad.at(layer).setZero();
    }
}

void NeuralNetwork::gradient_descent(const vector<DataPoint> &dataset, double dw = DEFAULT_LEARN_STEP, double db = DEFAULT_LEARN_STEP)
{
    for (int i = 0; i < dataset.size(); i++)
    {
        back_propagate(dataset.at(i));
    }
    update_parameters(dw, db);
}

void NeuralNetwork::stochastic_descent(const vector<DataPoint> &dataset, double dw = DEFAULT_LEARN_STEP, double db = DEFAULT_LEARN_STEP)
{
    for (int i = 0; i < dataset.size(); i++)
    {
        stochastic_descent(dataset.at(i), dw, db);
    }
}

void NeuralNetwork::stochastic_descent(const DataPoint &point, double dw = DEFAULT_LEARN_STEP, double db = DEFAULT_LEARN_STEP)
{
    back_propagate(point);
    update_parameters(dw, db);
}

void NeuralNetwork::batch_descent(const vector<DataPoint> &dataset, double dw = DEFAULT_LEARN_STEP, double db = DEFAULT_LEARN_STEP, int batch_size = 0)
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

void NeuralNetwork::momentum_descent(const vector<DataPoint> &dataset, double dw, double db, double gamma)
{
    back_propagate(dataset.at(0));
    vector<MatrixXd> prev_w_grad = w_grad;
    vector<MatrixXd> prev_b_grad = b_grad;
    int n_weights = weights.size();

    for (int i = 0; i < dataset.size(); i++)
    {
        back_propagate(dataset.at(i));
        for (int layer = 0; layer < n_weights; layer++)
        {
            prev_w_grad.at(layer) = prev_w_grad[layer] * gamma + w_grad[layer] * dw;
            prev_b_grad.at(layer) = prev_b_grad[layer] * gamma + b_grad[layer] * db;
            weights.at(layer) -= prev_w_grad[layer];
            biases.at(layer) -= prev_b_grad[layer];
            w_grad.at(layer).setZero();
            b_grad.at(layer).setZero();
        }
    }
}

void NeuralNetwork::nag_descent(const vector<DataPoint> &dataset, double dw, double db, double gamma)
{
    back_propagate(dataset.at(0));
    vector<MatrixXd> prev_w_grad = w_grad;
    vector<MatrixXd> prev_b_grad = b_grad;
    int n_weights = weights.size();

    for (int i = 0; i < dataset.size(); i++)
    {
        NeuralNetwork nn_cpy = *this;
        for (int layer = 0; layer < n_weights; layer++)
        {
            nn_cpy.weights.at(layer) -= prev_w_grad[layer] * gamma;
            nn_cpy.biases.at(layer) -= prev_b_grad[layer] * gamma;
        }
        nn_cpy.back_propagate(dataset[i]);

        for (int layer = 0; layer < n_weights; layer++)
        {
            prev_w_grad.at(layer) = prev_w_grad[layer] * gamma + nn_cpy.w_grad[layer] * dw;
            prev_b_grad.at(layer) = prev_b_grad[layer] * gamma + nn_cpy.b_grad[layer] * db;
            weights.at(layer) -= prev_w_grad[layer];
            biases.at(layer) -= prev_b_grad[layer];
            w_grad.at(layer).setZero();
            b_grad.at(layer).setZero();
        }
    }
}

void NeuralNetwork::adagrad_descent(const vector<DataPoint> &dataset, double eta, double epsilon)
{
    vector<double> grad_magnitude_sum(n_layers);
    int n_weights = weights.size();

    for (int i = 0; i < dataset.size(); i++)
    {
        back_propagate(dataset[i]);

        for (int layer = 0; layer < n_weights; layer++)
        {
            grad_magnitude_sum.at(layer) +=  w_grad[layer].norm() + b_grad[layer].norm();
            weights.at(layer) -= w_grad[layer] * eta / sqrt(grad_magnitude_sum.at(layer) + epsilon);
            biases.at(layer) -= b_grad[layer] * eta / sqrt(grad_magnitude_sum.at(layer) + epsilon);
            w_grad.at(layer).setZero();
            b_grad.at(layer).setZero();
        }
    }
}

void NeuralNetwork::adadelta_descent(const vector<DataPoint> &dataset, double eta, double e, double gamma)
{
    vector<double> grad_magnitude_sum(n_layers);
    int n_weights = weights.size();

    for (int i = 0; i < dataset.size(); i++)
    {
        back_propagate(dataset[i]);

        for (int layer = 0; layer < n_weights; layer++)
        {
            grad_magnitude_sum.at(layer) =  grad_magnitude_sum.at(layer) * gamma + (w_grad[layer].norm() + b_grad[layer].norm()) *(1-gamma);
            weights.at(layer) -= w_grad[layer] * eta / sqrt(grad_magnitude_sum.at(layer) + e);
            biases.at(layer) -= b_grad[layer] * eta / sqrt(grad_magnitude_sum.at(layer) + e);
            w_grad.at(layer).setZero();
            b_grad.at(layer).setZero();
        }
    }
}

void NeuralNetwork::adam_descent(const vector<DataPoint> &dataset, double b1, double b2, double e, double eta)
{
    vector<double> w_first_moment(n_layers);
    vector<double> b_first_moment(n_layers);
    vector<double> second_moment(n_layers);
    int n_weights = weights.size();

    for (int i = 0; i < dataset.size(); i++)
    {
        back_propagate(dataset[i]);

        for (int layer = 0; layer < n_weights; layer++)
        {
            w_first_moment.at(layer) = w_first_moment.at(layer) * b1 / (1-b1) + w_grad[layer].norm();
            b_first_moment.at(layer) = b_first_moment.at(layer) * b1 / (1-b1) +  b_grad[layer].norm();
            second_moment.at(layer) = second_moment.at(layer) * b2 / (1-b2) + (w_grad[layer].squaredNorm() + b_grad[layer].squaredNorm());
            weights.at(layer) -= w_grad[layer] * eta / (sqrt(second_moment.at(layer)) + e);
            biases.at(layer) -= b_grad[layer] * eta / (sqrt(second_moment.at(layer)) + e);
            w_grad.at(layer).setZero();
            b_grad.at(layer).setZero();
        }
    }
}

double NeuralNetwork::set_accuracy(const vector<DataPoint> &dataset)
{
    double correct = 0;
    for (DataPoint dp : dataset)
    {
        correct += (feed_forward(dp.input, true) == dp.label);
    }
    return correct / dataset.size();
}

double NeuralNetwork::set_cost(const vector<DataPoint> &dataset)
{
    double sum = 0;
    for (DataPoint dp : dataset)
    {
        sum += cost(feed_forward(dp.input), dp.label);
    }
    return sum;
}

void NeuralNetwork::evaluate(const vector<DataPoint> &dataset)
{
    std::cout << "=== Current Network Status ===\n";
    std::cout << "Weight matrix magnitude = " << weights[0].norm() << "\n";
    std::cout << "Dataset cost = " << set_cost(dataset) << "\n";
    std::cout << "Prediction accuracy = " << set_accuracy(dataset) << "\n";
}