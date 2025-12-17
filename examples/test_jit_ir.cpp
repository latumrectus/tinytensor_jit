#include <tt/tinytensor.h>
#include <iostream>

int main() {
    using namespace tinytensor;

    std::cout << "--- 1. Creating Graph ---\n";

    Tensor a = full(1.0f, {2, 2}, kJIT);
    Tensor b = full(2.0f, {2, 2}, kJIT);

    Tensor c = add(a, b);
    Tensor d = relu(c);

    std::cout << "--- 2. Triggering Compilation ---\n";

    d.eval();

    return 0;
}