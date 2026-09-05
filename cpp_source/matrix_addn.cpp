//
//  matrix_addn.cpp
//  
//
//  Created by Abhishek Chintada on 05/09/26.
// See test.cpp for execution standards.

#include <blaze/Blaze.h>
#include <iostream>
#include <cstddef>

void print_Vector(blaze::DynamicVector x) {
    std::cout << "[";
    for(size_t i {}, i <= 3, i++) {
        std::cout << x[i];
    }
    std::cout << "]";
}

int main(void) {
    blaze::DynamicVector<double> b {4.0, 5.0, 6.0};
    blaze::DynamicVector<double> a {1.0, 2.0, 3.0};
    blaze::DynamicVector<double> res(3, 0.0);
    blaze::DynamicMatrix<double, blaze::rowMajor> {{1.0, 2.0, 3.0},{4.0, 5.0, 6.0},{7.0, 8.0, 9.0}};
    blaze::DynamicVector<double> c {0.0, 0.0, 0.0};
    c = a + b;
    print_Vector(c);
    return 0;
}
