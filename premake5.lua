workspace "RL_Pokemon"
	
	architecture "x64"
	startproject "RL_Pokemon"

	configurations
	{
		"Debug",
		"Release"
	}

	flags 
	{
		"MultiProcessorCompile"
	}

--Output directories
outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

--Include directories (path relative to root folder)
IncludeDir={}
IncludeDir["Libtorch_Release"] = "RL_Pokemon/dependencies/libtorch/release/libtorch/include"
IncludeDir["Libtorch_torch_Release"] = "RL_Pokemon/dependencies/libtorch/release/libtorch/include/torch/csrc/api/include"
IncludeDir["Libtorch_Debug"] = "RL_Pokemon/dependencies/libtorch/debug/libtorch/include"
IncludeDir["Libtorch_torch_Debug"] = "RL_Pokemon/dependencies/libtorch/debug/libtorch/include/torch/csrc/api/include"

project "RL_Pokemon"
	location "RL_Pokemon"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "off"

	targetdir("bin/" ..outputdir.. "/%{prj.name}")
	objdir("bin-int/" ..outputdir.. "/%{prj.name}")

	pchheader "pch.h"
	pchsource "RL_Pokemon/src/pch.cpp"

	files
	{
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp"
	}

	includedirs
	{
		"RL_Pokemon/src",
	}

	

	filter "system:windows"
		systemversion "latest"

		filter "configurations:Debug"
			runtime "Debug"
			symbols "on"
			includedirs { "%{IncludeDir.Libtorch_Debug}", "%{IncludeDir.Libtorch_torch_Debug}" }
			libdirs { "RL_Pokemon/dependencies/libtorch/debug/libtorch/lib" }
			links{ "torch.lib", "caffe2_module_test_dynamic.lib", "c10.lib" }
		
		filter "configurations:Release"
			runtime "Release"
			optimize "on"
			includedirs { "%{IncludeDir.Libtorch_Release}", "%{IncludeDir.Libtorch_torch_Release}" }
			libdirs { "RL_Pokemon/dependencies/libtorch/release/libtorch/lib" }
			links{ "torch.lib", "caffe2_module_test_dynamic.lib", "c10.lib" }
