#pragma once

struct Experience
{
	Experience() = default;
	Experience(torch::Tensor& state, torch::Tensor& action,
		torch::Tensor& nextState, torch::Tensor& reward, torch::Tensor& done)
	{
		State = state;
		Action = action;
		NextState = nextState;
		Reward = reward;
		Done = done;
	}
	// probably implement a move constructor as we 
	// because the copying might be expensive

	torch::Tensor State;
	torch::Tensor Action;
	torch::Tensor NextState;
	torch::Tensor Reward;
	torch::Tensor Done; // flag to be set when an episode ends (battle ends most probably)
};

class ReplayMemory
{
public:
	ReplayMemory() = default;
	ReplayMemory(uint64_t capacity, uint32_t batchSize, uint32_t numStates, torch::Device device);

	void Push(Experience& exp); // or use std::move // overload
	Experience Sample(uint32_t batchSize);

	inline bool CanProvideSample(uint32_t batchSize) const { return batchSize <= _Memory.size(); }
	inline uint64_t GetMemorySize() const { return _Memory.size(); }

private:
	uint64_t _Capacity = 0;
	uint64_t _PushCount = 0;
	uint32_t _NumStates = 0;

	std::vector<Experience> _Memory;
	std::vector<int64_t> _RandomNumList;
	
	torch::Device _Device;
	std::mt19937 _Generator; // do not call this repeatedly
};
