//
//  matrix_addn.cpp
//  
//
//  Created by Abhishek Chintada on 05/09/26.
// See test.cpp for execution standards.

#include <blaze/Blaze.h>
#include <iostream>
#include <cstddef>

template<typename VectorType>
void print_Vector(const VectorType& x) {
    std::cout << "[ ";
    for(size_t i {}; i < 3; i++) {
        std::cout << x[i] << "  ";
    }
    std::cout << "]" << std::endl;
}

void print_Matrix(const blaze::DynamicMatrix<double, blaze::rowMajor>& x) {
    std::cout << "[" ;
    for(size_t i {}; i < x.rows(); i++) {
        if(i != 0) std::cout << " ";
        print_Vector(blaze::row(x, i));
    }
}

int main(void) {
    blaze::DynamicVector<double> b {4.0, 5.0, 6.0};
    blaze::DynamicVector<double> a {1.0, 2.0, 3.0};
    print_Vector(a);
    print_Vector(b);
    blaze::DynamicVector<double> res(3, 0.0);
    blaze::DynamicMatrix<double, blaze::rowMajor> Mat {{1.0, 2.0, 3.0},{4.0, 5.0, 6.0},{7.0, 8.0, 9.0}};
    print_Matrix(Mat);
    blaze::DynamicVector<double> c {0.0, 0.0, 0.0};
    res = a + b;
    print_Vector(res);
    c = Mat * a;
    print_Vector(c);
    return 0;
}
