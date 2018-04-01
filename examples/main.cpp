#include <iostream>
#include <vector>

int basics(int size, const int *a, const int *b, int *c);

int testVirtual(int size, int *c, int val);
int testNonVirtual(int size, int *c, int val);
int testLambda(int size, int *c, int val);

int main() {
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	//basics(arraySize, a, b, c);
	//testVirtual(arraySize, c, 10);
	//testNonVirtual(arraySize, c, 10);
	testLambda(arraySize, c, 17);

	std::cout << "Resulting vector C:";
	for (int idx = 0; idx < arraySize; idx++) {
		std::cout << ' ' << c[idx];
	}
	std::cout << '\n';
}
