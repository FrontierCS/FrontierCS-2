#include <spectrum/common/random.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_elements> <exponent> <N>\n";
        return 1;
    }

    long num_elements = std::stol(argv[1]);
    double exponent = std::stod(argv[2]);
    int N = std::stoi(argv[3]);

    if (num_elements <= 0 || N <= 0) {
        std::cerr << "Both num_elements and N should be positive.\n";
        return 1;
    }

    spectrum::Zipf zipf(num_elements, exponent);

    for (int i = 0; i < N; ++i) {
        std::cout << zipf.Next() << "\n";
    }

    return 0;
}