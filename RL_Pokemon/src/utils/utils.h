#pragma once

#include "dqn/dqnet.h"

namespace utils {

	void LoadStateDict(Dqnet& model, Dqnet& targetModel); // copy parameters from one model to another

}

