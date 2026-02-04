/**
 * CS3210 Tutorial 1 - Exercise 15
 * Original asdf.cpp with command line argument support for benchmarking
 *
 * Algorithm: Bubble Sort - O(n²) time complexity
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

std::vector<int> generateLargeArray(size_t size, size_t max_value) {
    std::vector<int> array(size);
    for (size_t i = 0; i < size; ++i) {
        array[i] = rand() % max_value;
    }
    return array;
}

// Original Bubble Sort - O(n²)
void my_sort(std::vector<int>& array) {
    size_t n = array.size();
    for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = 0; j < n - i - 1; ++j) {
            if (array[j] > array[j + 1]) {
                int temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Allow command line argument for array size
    size_t ARRAY_SIZE = 10000;
    if (argc > 1) {
        ARRAY_SIZE = std::atoi(argv[1]);
    }
    const size_t MAX_VALUE = 100000;

    std::vector<int> largeArray = generateLargeArray(ARRAY_SIZE, MAX_VALUE);

    my_sort(largeArray);

    // Print the last 5 elements of the sorted array
    std::cout << "Last 5 elements of the sorted array: ";
    for (size_t i = ARRAY_SIZE - 5; i < ARRAY_SIZE; ++i) {
        std::cout << largeArray[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
