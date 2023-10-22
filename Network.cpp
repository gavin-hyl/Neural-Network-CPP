#include <cmath>
#include <vector>
#include <algorithm>
#include "Network.h"
#include "Functions.h"

#define DEFAULT_LEARN_STEP 0.001

using std::vector;

class DataPoint {
    public:
        Matrix data;
        Matrix expected;
        DataPoint(Matrix data, Matrix expected) {
            this->data = data;
            this->expected = expected;
        }
};

class NeuralNetwork {

    private:
        vector<Matrix> weights;             // all weights between layers, n-1 in total
        vector<Matrix> biases;              // all biases between layers, n-1 in total
        vector<Matrix> gradientWeightCost;  // intermediate variable, used to pass the results from back propagation to gradient descent
        vector<Matrix> gradientBiasCost;    // intermediate variable, used to pass the results from back propagation to gradient descent
        vector<Matrix> layerValues;         // includes values of all layers, including the input and output, n in total
        int nLayers;                        // total number of layers

        // cost function
        double cost(Matrix& output, Matrix& expected) {
            return pow((output-expected).abs(), 2);
        }

        // cost function's derivative
        Matrix gradOutputCost(Matrix output, Matrix expected) {
            return (output - expected) * 2;
        }

    public:
        // Constructor
        NeuralNetwork(vector<int> layerSizes) {
            nLayers = layerSizes.size();
            for (int i = 0; i < nLayers - 1; i++) {
                // note that there are only nLayers-1 layers of weights and biases
                weights.push_back(Matrix(layerSizes[i+1], layerSizes[i], RANDOM));
                biases.push_back(Matrix(layerSizes[i+1], layerSizes[i], RANDOM));
                gradientWeightCost.push_back(Matrix(layerSizes[i+1], layerSizes[i]));
                gradientBiasCost.push_back(Matrix(layerSizes[i+1], layerSizes[i]));
            }
            layerValues.reserve(nLayers);
        }

        /**
         * @brief Calculate the output of the network given a single input
         * 
         * @param input 
         * @param getMax if true, returns a matrix with the highest probability index set to 1 and all others set to zero
         * @param recordLayerValues if true, records the layer values internally
         * @return Matrix 
         */
        Matrix feedforward(Matrix& input, bool getMax=false, bool recordLayerValues=false) {
            if (recordLayerValues) {
                for (int i = 0; i < nLayers - 1; i++) {
                    layerValues[i] = input;
                    input = sigma(weights[i] * input + biases[i]);
                }
            } else {
                for (int i = 0; i < nLayers - 1; i++) {
                    input = sigma(weights[i]*input + biases[i]);
                }
            }
            Matrix output = softmax(input);
            layerValues[nLayers] = output.dup();
            if (getMax) {
                vector<double> outArray = output.getVectorCol(0);
                int maxIdx = std::max_element(outArray.begin(), outArray.end()) - outArray.begin();
                output = Matrix(outArray.size(), 1);
                output.elements[maxIdx][0] = 1; // all other elements are 0
                return output;
            }
            return output;
        }
        
        /**
         * @brief Calculate the gradient of the cost function at one data point
         * 
         * @param point the data point
         */
        void backPropagate(DataPoint point) {
            Matrix deltaL = gradOutputCost(feedforward(point.data, false, true), point.expected);
            for (int layer = nLayers-2; layer >= 0; layer--) {
                // layer=0 corresponds to the first layer and the weights from the first to second layer
                gradientWeightCost[layer] = deltaL * layerValues[layer].T();
                gradientBiasCost[layer] = deltaL;
                deltaL = sigmaP(weights[layer].T() * deltaL);
            }
        }

        /**
         * @brief Vanilla gradient descent algorithm, using the gradient calculated by backPropagate
         * 
         * @param learnStep step size
         */
        void gradientDescent(double learnStep=DEFAULT_LEARN_STEP) {
            for (int i = 0; i < nLayers; i++) {
                weights[i] = weights[i] - gradientWeightCost[i] * learnStep;
                biases[i] = biases[i] - gradientBiasCost[i] * learnStep;
            }
        }
};