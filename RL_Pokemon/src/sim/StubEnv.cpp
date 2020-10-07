#include "pch.h"
#include "StubEnv.h"

StubEnv::StubEnv(uint32_t numStates, torch::Device device)
	:_NumStates(numStates), _Device(device)
{
	_State  = torch::ones({ 1, numStates }, _Device);
	_Reward = torch::ones({ 1, 1 }, _Device);
	_Done = false;
	torch::Tensor _TestVar = torch::ones({ 1, numStates }, _Device) * 5;
}

void StubEnv::ResetEnv()
{
	count = 0;
}

void StubEnv::TakeAction(torch::Tensor& action)
{
	count++;
	if (count >= 15)
		_Done = true;
}

