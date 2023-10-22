#include "Matrix.h"
#include "Network.cpp"
#include <cmath>

using std::vector;

int main(int argc, char const *argv[])
{
    // tests to see if the model can classify points inside the unit circle correctly.
    int nSamples = 1000;
    vector<DataPoint> dataset;
    double x, y;
    bool inCircle;
    for (int i = 0; i < nSamples; i++) {
        x = 2.0 * rand()/RAND_MAX;
        y = 2.0 * rand()/RAND_MAX;
        inCircle = (x*x + y*y) < 1;
        vector<double> input (x, y);
        Matrix in = Matrix::toMatrix(input, 2, 1);
        dataset.push_back(DataPoint(in, Matrix::toMatrix(inCircle)));
    }

    vector<int> layers {2, 3, 3, 1};
    NeuralNetwork network = NeuralNetwork(layers);
    return 0;
}
