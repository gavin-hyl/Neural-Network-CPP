#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <random>

using std::vector;

enum { INDEX = -1, SIZE = -2 };
enum { ZEROES, IDEN, RANDOM, FUNC };

class Matrix 
{
    public:
        int rows = 1;
        int cols = 1;
        
        vector<vector<double>> elements;
        Matrix(int r = 1, int c = 1, int type = ZEROES, double (*gen)(int, int) = NULL);
        
        Matrix operator + (Matrix m);
        Matrix operator - (Matrix m);
        Matrix operator * (Matrix m);
        Matrix operator * (double c);
        Matrix operator ^ (int pow);
        double operator & ();
        bool operator == (Matrix m);
        friend std::ostream& operator << (std::ostream& out, const Matrix& M);
    
        bool eq_size(Matrix m);
        static Matrix toMatrix(vector<double>, int r, int c);
        static Matrix toMatrix(double e);
        vector<double> getRow(int r);
        vector<double> getVectorCol(int c);
        bool is_sqr();
        bool valid_mult(Matrix m);
        double abs();
        Matrix inv();
        double det();
        Matrix dup();
        Matrix T();
};