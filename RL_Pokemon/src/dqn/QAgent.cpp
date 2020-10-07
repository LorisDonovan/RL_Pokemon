#include "pch.h"
#include "QAgent.h"

QAgent::QAgent(uint32_t numActions, torch::Device device, float start, float end, float decayRate)
	:_NumActions(numActions), _Device(device), 
	_Generator((std::random_device())()), _RandomNum(0, 1),
	_Start(start), _End(end), _DecayRate(decayRate)
{
	_CurrentStep = 0;
}

torch::Tensor QAgent::SelectAction(torch::Tensor& state, Dqnet policyNet)
{
	float epsilon = GetExplorationRate();
	_CurrentStep++;
	//std::cout << "Exploration Rate = " << epsilon << "\tCurrent Step = " << _CurrentStep << std::endl;
	if (epsilon > _RandomNum(_Generator)) // explore
	{
		torch::Tensor action = torch::randint(0, _NumActions, {1}, _Device);
		//std::cout << "Random action = " << action.item<int32_t>() << std::endl;
		return action;
	}
	else // exploit
	{
		torch::NoGradGuard noGrad;
		torch::Tensor action = policyNet->forward(state).argmax(1).to(_Device);
		//std::cout << "Policy action = " << action.item<int32_t>() << std::endl;
		return action;
	}

}
