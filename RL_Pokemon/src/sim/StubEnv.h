#pragma once

class StubEnv
{
public:
	StubEnv(uint32_t numStates, torch::Device device);

	void TakeAction(torch::Tensor& action);
	void ResetEnv();

	inline torch::Tensor GetState() const { return _State; }
	inline torch::Tensor GetReward() const { return _Reward; }
	inline bool GetDone() const { return _Done; }

private:
	uint32_t _NumStates;
	torch::Device _Device;
	uint32_t count = 0;

	torch::Tensor _State;
	torch::Tensor _Reward;
	bool _Done;

	torch::Tensor _TestVar;
};
