/*
* Codigo realizado por Hugo Montañés García.
*/

#pragma once

#include <string>
#include <map>
#include "document.h"
#include "prettywriter.h"
#include "filewritestream.h"
#include "filereadstream.h"

namespace space
{
	class ShaderParameterLogger
	{
	private:
		std::string jsonFilePath;
		rapidjson::Document document;

		//Load existing JSON file or create new one if it does not exist
		bool loadJsonFile();

	public:

		ShaderParameterLogger(const std::string& filepath = "../../../assets/database_images/tags.json");
		~ShaderParameterLogger() = default;

		//Log parameters for a specific image
		bool logParameters(
			const std::string& imageName, 
			const std::map<std::string, std::string>& tags, 
			const std::map<std::string, float>& parameters);


		//Save the current JSON document to file
		bool saveJsonFile();
	};
}