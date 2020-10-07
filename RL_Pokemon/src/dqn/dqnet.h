#pragma once

class DqnetImpl : public torch::nn::Module
{
public:
	DqnetImpl(uint32_t numStates, uint32_t numActions);
	torch::Tensor forward(torch::Tensor x);

private:
	torch::nn::Linear _fc1;
	torch::nn::Linear _fc2;
	torch::nn::Linear _fc3;
	torch::nn::Linear _out;
};

TORCH_MODULE(Dqnet);
