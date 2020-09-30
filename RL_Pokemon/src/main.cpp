#include "pch.h"

int main(int argc, char* argv[])
{
	torch::Tensor input = torch::randn({ 1,2 });
	std::cout << input << std::endl;
	std::cin.get();
	return 0;
}