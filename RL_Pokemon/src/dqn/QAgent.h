#pragma once

#include "dqnet.h"

class QAgent
{
public:
	QAgent(uint32_t numActions, torch::Device device = torch::kCPU, float start = 1.0f, float end = 0.01f, float decayRate = 0.001f);

	torch::Tensor SelectAction(torch::Tensor& state, Dqnet policyNet);

private:
	inline float GetExplorationRate() const { return _End + (_Start - _End) * std::exp(-1 * _CurrentStep * _DecayRate); }

private:
	uint32_t _NumActions;
	int64_t _CurrentStep;
	torch::Device _Device;

	// Random Number Generator
	std::mt19937 _Generator;
	std::uniform_real_distribution<double> _RandomNum; // generate random num [0, 1) from uniform distribution
	
	// for epsilon greedy strategy
	float _Start, _End, _DecayRate;
};
