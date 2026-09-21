#include "blaze_essentials.hpp"
#include <random>
#include <chrono>

int main(void) {
    // blaze implementation
    blaze::setNumThreads(8);
    auto start_blaze = std::chrono::high_resolution_clock::now();
    size_t N = 10000000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(1, 20);
    blaze::DynamicVector<double, blaze::columnVector> a(N), b(N), c(N);
    for(size_t i {}; i < N; i++) {
        a[i] = distribution(gen);
        b[i] = distribution(gen);
    }
    auto end_blaze = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_blaze = end_blaze - start_blaze;
    // print_Vector(a, N);
    // print_Vector(b, N);
    c = a*b;
    // print_Vector(c, N);
    std::cout << "<console> Duration for blaze : " << elapsed_blaze.count() << std::endl;
    


    // normal std vector implementation
    auto start_vec = std::chrono::high_resolution_clock::now();
    std::vector<double> v1(N, 0);
    std::vector<double> v2(N, 0);
    std::vector<double> sum(N, 0);
    for(size_t i {}; i < N; i++) {
        v1[i] = distribution(gen);
        v2[i] = distribution(gen);
        sum[i] = v1[i] + v2[i];
    }
    auto end_vec = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_vec = end_vec - start_vec;
    std::cout << "<console> Duration for vector : " << elapsed_vec.count() << std::endl;

    return 0;
}