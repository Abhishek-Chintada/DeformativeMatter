#include <iostream>
#include <random>
#include <blaze/Blaze.h>

int main(void) {
    size_t N = 10;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(1, 20);
    blaze::DynamicVector<double, blaze::columnVector> a(N), b(N), c(N);
    for(size_t i {}; i < N; i++) {
        a[i] = distribution(gen);
        b[i] = distribution(gen);
    }
    c = a+b;
    return 0;
}