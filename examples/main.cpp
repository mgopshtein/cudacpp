#include <iostream>
#include <vector>

int basics(int size, const int *a, const int *b, int *c);

int main() {
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	basics(arraySize, a, b, c);
}
