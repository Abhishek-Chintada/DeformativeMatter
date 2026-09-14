#include <blaze/Blaze.h>
#include <iostream>

void printVector(const blaze::DynamicVector<double>& x) {
    
}

int main(void) {
    blaze::setNumThreads(8);
    size_t size = 3;
    blaze::DynamicVector<double, blaze::columnVector> a {1, 2, 3};
    blaze::DynamicMatrix<double> mat {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::cout << "[";
    for(size_t i {}; i < size; i++) {
        if(i != size-1) std::cout << a[i] << "  ";
        else std::cout << a[i] << "]" << std::endl;
    }
    std::cout << "[";
    for(size_t i {}; i < size; i++) {
        if(i==0) std::cout << "[" ;
        else std::cout << " [";
        for(size_t j {}; j < size; j++) {
            if(j != size-1) std::cout << mat(i, j) << "  ";
            else std::cout << mat(i, j) << "]";
        }
        if(i != size-1) std::cout << std::endl;
        else std::cout << "]" << std::endl;
    }
}
