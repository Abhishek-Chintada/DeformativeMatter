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


