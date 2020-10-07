#include "pch.h"
#include "dqn/dqnet.h"
#include "dqn/QAgent.h"
#include "dqn/ReplayMemory.h"

int main(int argc, char* argv[])
{
	torch::Device device = torch::kCUDA;
	uint32_t numActions = 4;
	uint32_t numStates = 6;
	uint32_t batchSize = 4;
	uint32_t capacity = 8;

	Dqnet net(numStates, numActions);
	net->to(device);
	QAgent agent(4, device);
	ReplayMemory mem(capacity, batchSize, numStates, device);

	torch::Tensor state     = torch::randn({ 1, numStates }, device);
	torch::Tensor action    = torch::zeros({ 1, 1 }, torch::kInt64).to(device); // has to be int64 for torch::gather() index
	torch::Tensor nextState = torch::randn({ 1, numStates }, device);
	torch::Tensor reward    = torch::randn({ 1, 1 }, device);
	torch::Tensor done      = torch::ones({ 1, 1 }, torch::kInt8).to(device);

	for (int32_t i = 0; i < 10; i++)
	{
		action = agent.SelectAction(state, net);
		Experience exp(state, action, nextState, reward, done);
		mem.Push(exp);
	}

	Experience exp;
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
