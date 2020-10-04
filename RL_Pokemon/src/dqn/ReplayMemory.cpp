#include "pch.h"
#include "ReplayMemory.h"


ReplayMemory::ReplayMemory(uint64_t capacity, uint32_t batchSize, uint32_t numStates, torch::Device device)
	:_Capacity(capacity), _NumStates(numStates), _Device(device), _Generator((std::random_device())())
{
}

void ReplayMemory::Push(Experience& exp)
{
	if (_Memory.size() < _Capacity)
	{
		_Memory.push_back(exp);
		_RandomNumList.push_back(_PushCount);
	}
	else
		_Memory.insert(_Memory.begin() + (_PushCount % _Capacity), exp);
	
	std::cout << "Pushed to: " << _PushCount % _Capacity << std::endl;
	_PushCount++;
}

Experience ReplayMemory::Sample(uint32_t batchSize)
{
	torch::Tensor states	 = torch::zeros({ (int32_t)batchSize, _NumStates }, _Device);
	torch::Tensor actions	 = torch::zeros({ (int32_t)batchSize, 1 }, torch::kInt64).to(_Device); // has to be int64 for torch::gather() index
	torch::Tensor nextStates = torch::zeros({ (int32_t)batchSize, _NumStates }, _Device);
	torch::Tensor rewards	 = torch::zeros({ (int32_t)batchSize, 1 }, _Device);
	torch::Tensor done		 = torch::zeros({ (int32_t)batchSize, 1 }, torch::kInt8).to(_Device);

	std::shuffle(_RandomNumList.begin(), _RandomNumList.end(), _Generator); // or use std::random_shuffle // but this was removed in C++17
	
	for (int32_t i = 0; i < batchSize; i++)
	{
		int32_t index = _RandomNumList[i];

		states[i]		= _Memory[index].State[0];
		actions[i]		= _Memory[index].Action[0];
		nextStates[i]	= _Memory[index].NextState[0];
		rewards[i]		= _Memory[index].Reward[0];
		done[i]			= _Memory[index].Done[0];
	}
	
	return Experience(states, actions, nextStates, rewards, done);
}
