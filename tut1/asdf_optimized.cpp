/**
 * CS3210 Tutorial 1 - Exercise 15
 * Optimized version of asdf.cpp
 *
 * Optimization: Replace O(n²) Bubble Sort with O(n log n) std::sort
 * std::sort uses Introsort (hybrid of quicksort, heapsort, and insertion sort)
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>  // For std::sort

std::vector<int> generateLargeArray(size_t size, size_t max_value) {
    std::vector<int> array(size);
    for (size_t i = 0; i < size; ++i) {
        array[i] = rand() % max_value;
    }
    return array;
}

// Optimized sort using std::sort (Introsort - O(n log n))
void my_sort(std::vector<int>& array) {
    std::sort(array.begin(), array.end());
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
