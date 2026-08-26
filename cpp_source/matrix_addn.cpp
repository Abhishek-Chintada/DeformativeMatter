#include <blaze/Blaze.h>
#include <cstdio>
#include <random>
#include <omp.h>
using namespace blaze;


int main(void) {
    omp_set_num_threads(4); // limits the thread usage to 4, data may get cold -> RAM gets hot!
    size_t N = 10;
    DynamicVector<double, columnVector> a(N), b(N), c(N);
    std::random_device dev;
    std::mt19937 gen_number(dev());
    std::uniform_real_distribution<double>distri(1, 50);
    for(size_t i = 0; i < N; i++) {
        a[i] = distri(gen_number);
        b[i] = distri(gen_number);
        c[i] = 0;
    }
    for(size_t i = 0; i < N; i++) {
        printf("a[%zu] = %lf; b[%zu] = %lf; c[%zu] = %lf;\n", i, a[i], i, b[i], i, c[i]);
    }
    c = a + b;
    printf("\n\nThe sum operation is completed at this statement.\n\n");
    for(size_t i = 0; i < N; i++) {
        printf("a[%zu] = %lf; b[%zu] = %lf; c[%zu] = %lf;\n", i, a[i], i, b[i], i, c[i]);
    }
    printf("The program is completed.\n");
}
