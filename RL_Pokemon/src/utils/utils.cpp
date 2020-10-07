#include "pch.h"
#include "utils.h"

namespace utils {

	void LoadStateDict(Dqnet& model, Dqnet& targetModel)
	{
		torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
		auto new_params = targetModel->named_parameters();
		auto params = model->named_parameters(true);

		for (auto& val : new_params)
		{
			auto name = val.key();
			auto* t = params.find(name);
			if (t != nullptr)
			{
				t->copy_(val.value().set_requires_grad(false));
			}
		}
		torch::autograd::GradMode::set_enabled(true);
	}

}
