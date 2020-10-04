#include "pch.h"
#include "dqn/ReplayMemory.h"

int main(int argc, char* argv[])
{
	uint64_t capacity = 4;
	uint32_t batchSize = 2;
	uint32_t numStates = 1;
	torch::Device device = torch::kCUDA;
	ReplayMemory mem(capacity, batchSize, numStates, device);

	torch::Tensor state		= torch::randn({ 1, numStates }, device);
	torch::Tensor action	= torch::ones({ 1, 1 }, torch::kInt64).to(device); // has to be int64 for torch::gather() index
	torch::Tensor nextState	= torch::randn({ 1, numStates }, device);
	torch::Tensor reward	= torch::randn({ 1, 1 }, device);
	torch::Tensor done		= torch::ones({ 1, 1 }, torch::kInt8).to(device);

	Experience exp(state, action, nextState, reward, done);
	
	mem.Push(exp);
	mem.Push(exp);
	mem.Push(exp);
	mem.Push(exp);
	mem.Push(exp);
	mem.Push(exp);

	if (mem.CanProvideSample(batchSize))
		exp = mem.Sample(batchSize);

	std::cout << "Sampling..." << std::endl;
	std::cout << "State: \n" << exp.State << std::endl;
	std::cout << "Action: \n" << exp.Action << std::endl;
	std::cout << "NextState: \n" << exp.NextState << std::endl;
	std::cout << "Reward: \n" << exp.Reward << std::endl;
	std::cout << "Done: \n" << exp.Done << std::endl;


	std::cin.get();
	return 0;
}
