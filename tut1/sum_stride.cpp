#include <iostream>
#include <vector>
#include <cstdlib>

std::vector<int> generateLargeArray(size_t size) {
    std::vector<int> array(size);
    for (size_t i = 0; i < size; ++i) {
        array[i] = rand() % 100;
    }
    return array;
}

long long sumArrayWithStrides(const std::vector<int>& array, size_t stride) {
    long long sum = 0;
    for (size_t j = 0; j < stride; j++) {
        for (size_t i = j; i < array.size(); i += stride) {
            sum += array[i];
        }
    }
    return sum;
}

int main(int argc, char* argv[]) {
    const size_t ARRAY_SIZE = 100000000;
    size_t stride = (argc > 1) ? std::atoi(argv[1]) : 1;
    
    std::vector<int> largeArray = generateLargeArray(ARRAY_SIZE);
    long long totalSum = sumArrayWithStrides(largeArray, stride);
    std::cout << "Stride: " << stride << ", Sum: " << totalSum << "\n";
    return 0;
}