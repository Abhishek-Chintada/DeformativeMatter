#pragma once

#include <blaze/Blaze.h>
#include <iostream>


template<typename VectorType>
void print_Vector(const VectorType& x, size_t N) {
    std::cout << "[ ";
    for(size_t i {}; i < N; i++) {
        std::cout << x[i] << "  ";
    }
    std::cout << "]" << std::endl;
}


void print_Matrix(const blaze::DynamicMatrix<double, blaze::rowMajor>& x, size_t N) {
    std::cout << "[" ;
    for(size_t i {}; i < x.rows(); i++) {
        if(i != 0) std::cout << " ";
        print_Vector(blaze::row(x, i), N);
    }
}

