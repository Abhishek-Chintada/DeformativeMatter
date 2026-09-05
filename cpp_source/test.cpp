#include <blaze/Blaze.h> // OS - Linux/Debian & MacOS
#include <iostream>

int main(void) {
    std::cout << "Notlo Modda" << std::endl;
    return 0;
}

/* Apple Compiler:
/opt/homebrew/opt/llvm/bin/clang++ \
  -std=c++20 \
  -O3 \
  -march=native \
  -I/opt/homebrew/include \
  -I/opt/homebrew/opt/libomp/include \
  -L/opt/homebrew/opt/libomp/lib \
  -fopenmp \
  test.cpp \
  -o executable \
  -framework Accelerate
*/

// ----------------------------// 

/*Linux(Debian) Compiler:
clang++ -std=c++20 -O3 -march=native -fopenmp -lblas test.cpp -o executable
 */
