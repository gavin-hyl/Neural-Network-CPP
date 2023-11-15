#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <ctime>
#include "Matrix.h"

using std::vector;

Matrix::Matrix(int r, int c, int type, double (*gen)(int , int)) {
    if (c == 0) {
        c = r;
        type = IDEN;
    }
    if (r <= 0 || c <= 0) { throw INDEX; }
    rows = r;
    cols = c;
    if (type == ZEROES) {
        while(r-- > 0) {
            vector<double> row(c);
            elements.push_back(row);
        }
    }
    else if (type == IDEN) {
        while(r-- > 0) {
            vector<double> row(c);
            row[c-r-1] = 1;
            elements.push_back(row);
        }
    }
    else if (type == RANDOM) {
        srand((int) time(0));
        while(r-- > 0) {
            vector<double> row(c);
            for (double& e : row){
                e = (1.0*rand()/RAND_MAX) - 0.5;
            }
            elements.push_back(row);
        }
    }
    else {
        for (int i = 0; i < r; i++) {
            vector<double> row(c);
            for (int j = 0; j < c; j++) {
                row[j] = gen(i, j);
            }
        elements.push_back(row);
        }
    }
}


bool Matrix::eq_size(Matrix m) { return cols == m.cols && rows == m.rows; }
bool Matrix::is_sqr() {return cols == rows; }
bool Matrix::valid_mult(Matrix m) { return cols == m.rows; }

Matrix Matrix::operator + (const Matrix& m) const {
    Matrix result = Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.elements[i][j] = elements[i][j] + m.elements[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator - (const Matrix& m) const {
    Matrix result = Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.elements[i][j] = elements[i][j] - m.elements[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator * (const Matrix& m) const {
    int c = m.cols;
    Matrix result = Matrix(rows, c);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < cols; k++) {
                result.elements[i][j] += elements[i][k] * m.elements[k][j];
            }
        }
    }
    return result;
}

Matrix Matrix::operator * (const double& c) const {
    Matrix result = Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.elements[i][j] = elements[i][j] * c;
        }
    }
    return result;
}

Matrix Matrix::operator ^ (int pow) {
    Matrix iden = Matrix(rows, cols, IDEN);
    if (pow == 0) {
        return iden;
    }
    else if (pow > 0) {
        while (pow-- > 0) {
            iden = iden * *this;
        }
        return iden;
    } 
    else {

    }
}

bool Matrix::operator == (Matrix m) {
    if (!eq_size(m)) { return false; }
    Matrix result = Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (elements[i][j] != m.elements[i][j]) { return false; }
        }
    }
    return true;
}

std::ostream& operator << (std::ostream& out, const Matrix& m) {
    int r = m.rows;
    int c = m.cols;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            out << m.elements[i][j] << " ";
        }
        out << std::endl;
    }
    return out << "-----\n";
}

double &Matrix::operator() (int r, int c)
{
    return elements.at(r).at(c);
}

void Matrix::print() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << elements[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-----\n";
}

void Matrix::clear() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->elements.at(i).at(j) = 0;
        }
    }
}

Matrix Matrix::T() {
    Matrix transpose = Matrix(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transpose.elements[j][i] = elements[i][j];
        }
    }
    return transpose;
}

// column major
// Matrix Matrix::toMatrix(vector<double> elements, int r, int c) {
//     Matrix M = Matrix(r, c);
//     int size = elements.size();
//     int idx = 0;
//     for (int i = 0; i < c; i++) {
//         for (int j = 0; j < r; j++) {
//             int idx = i*r + c;
//             if (idx >= size) {
//                 return M;  // the others were filled with 0 anyway
//             }
//             M.elements[j][i] = elements[idx];
//         }
//     }
//     return M;
// }

Matrix Matrix::toMatrix(vector<double> elements) {
    int d = elements.size();
    Matrix M = Matrix(d, 1);
    for (int i = 0; i < d; i++) {
        M.elements[i][0] = elements[i];
    }
    return M;
} 

Matrix Matrix::toMatrix(double e) {
    Matrix M = Matrix(1, 1);
    M.elements[0][0] = e;
    return M;
}

Matrix Matrix::toBasis(int idx, int dim) {
    Matrix V = Matrix(dim, 1);
    V.elements[idx][0] = 1;
    return V;
}

vector<double> Matrix::getRow(int r) {
    vector<double> row;
    for (int i = 0; i < cols; i++) {
        row.push_back(this->elements[r][i]);
    }
    return row;
}

vector<double> Matrix::getVectorCol(int c) {
    vector<double> col;
    for (int i = 0; i < rows; i++) {
        col.push_back(elements[i][c]);
    }
    return col;
}

double Matrix::abs() {
    double result = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result += elements[i][j] * elements[i][j];
        }
    }
    return result;
}

Matrix Matrix::dup() {
    Matrix duplicate = Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            duplicate.elements[i][j] = elements[i][j];
        }
    }
    return duplicate;
}

Matrix Matrix::schur(const Matrix& M) {
    Matrix result = Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.elements[i][j] = this->elements[i][j] * M.elements[i][j];
        }
    }
    return result;
}