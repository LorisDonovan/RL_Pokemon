#include "pch.h"
#include "dqnet.h"

DqnetImpl::DqnetImpl(uint32_t numStates, uint32_t numActions)
	:_fc1(torch::nn::LinearOptions(numStates, 256)), 
	_fc2(torch::nn::LinearOptions(256, 128)),
	_fc3(torch::nn::LinearOptions(128, 64)),
	_out(torch::nn::LinearOptions(64, numActions))
{
	register_module("fc1", _fc1);
	register_module("fc2", _fc2);
	register_module("fc3", _fc3);
	register_module("out", _out);
}

torch::Tensor DqnetImpl::forward(torch::Tensor x)
{
	x = torch::relu(_fc1(x));
	x = torch::relu(_fc2(x));
	x = torch::relu(_fc3(x));
	x = _out(x);
	return x;
}
