#include "pch.h"
#include "dqn/dqnet.h"
#include "dqn/QAgent.h"
#include "dqn/ReplayMemory.h"
#include "utils/utils.h"

#include "sim/StubEnv.h"

int main(int argc, char* argv[])
{
	torch::Device device = torch::kCUDA;
	uint32_t numActions = 4;
	uint32_t numStates = 6;
	uint32_t batchSize = 4;
	uint32_t capacity = 8;

	float lr = 0.01;
	float gamma = 0.9;

	ReplayMemory mem(capacity, batchSize, numStates, device);
	QAgent agent(4, device);

	Dqnet policyNet(numStates, numActions);
	Dqnet targetNet(numStates, numActions);

	utils::LoadStateDict(policyNet, targetNet);
	policyNet->to(device);
	targetNet->to(device);
	targetNet->eval();

	StubEnv env(numStates, device);
	
	torch::Tensor state;
	torch::Tensor action;
	torch::Tensor nextState;
	torch::Tensor reward;
	bool done;
	torch::Tensor doneTensor;

	torch::optim::RMSprop optimizer(policyNet->parameters(), torch::optim::RMSpropOptions(lr));
	torch::Tensor loss;

	for (int32_t episode = 0; episode < 10; episode++)
	{
		env.ResetEnv();
		state = env.GetState();
		for (int32_t t = 0; ; t++)
		{
			action = agent.SelectAction(state, policyNet);
			env.TakeAction(action);
			nextState = env.GetState();
			reward = env.GetReward();
			done = env.GetDone();
			doneTensor = torch::tensor(done ? 0 : 1, device).reshape({ 1, 1 });

			Experience exp(state, action, nextState, reward, doneTensor);
			mem.Push(exp);

			if (mem.CanProvideSample(batchSize))
			{
				exp = mem.Sample(batchSize);

				// we take the predicted Q-value of the action we performed (hence .gather())
				torch::Tensor qValue = policyNet->forward(exp.State).gather(1, exp.Action); // predicted qValue
				torch::autograd::GradMode::set_enabled(false); // grad not needed for loss calc
				torch::Tensor maxQ = targetNet->forward(exp.NextState).max_values(1);
				torch::Tensor qTarget = exp.Reward + gamma * maxQ * exp.Done;
				// if next state terminates, qValue = reward(because done = 0)
				torch::autograd::GradMode::set_enabled(true);
				
				loss = torch::smooth_l1_loss(qValue, qTarget);
				policyNet->zero_grad();
				loss.backward();
				optimizer.step();

				std::cout << "Loss = " << loss << std::endl;
			}

			if ((t % 5) == 0)
			{
				utils::LoadStateDict(policyNet, targetNet);
			}

			if (done)
				break;
		}
	}
	
	std::cin.get();
	return 0;
}

