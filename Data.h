#pragma once
#include "Eigen/Dense"
#include <cmath>
#include <vector>

using std::vector;

typedef struct DataPoint_t 
{
    Eigen::VectorXd input;
    Eigen::VectorXd label;
} DataPoint;