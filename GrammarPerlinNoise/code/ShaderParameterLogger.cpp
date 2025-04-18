#include "ShaderParameterLogger.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include "istreamwrapper.h"
#include "ostreamwrapper.h"
#include "writer.h"

namespace space
{
	ShaderParameterLogger::ShaderParameterLogger(const std::string& filePath) : jsonFilePath(filePath)
	{
		//Initialize with empty object if file can't be loaded
		document.SetObject();
		loadJsonFile();
	}

	bool ShaderParameterLogger::loadJsonFile()
	{
		//Check if file exists 
		std::ifstream fileCheck(jsonFilePath);
		if (!fileCheck.good())
		{
			std::cout << "JSON file not found, will create a new one on save." << std::endl;
			return false;
		}
		fileCheck.close();

		//Open file for reading
		FILE* fp = fopen(jsonFilePath.c_str(), "rb");
		if (!fp)
		{
			std::cerr << "Error opening JSON file: " << jsonFilePath << std::endl;
			return false;
		}
		
		std::vector<char> readBuffer(65536);
		rapidjson::FileReadStream is(fp, readBuffer.data(), readBuffer.size());

		document.ParseStream(is);
		fclose(fp);

		if (document.HasParseError())
		{
			std::cerr << "Error parsing JSON file: " << document.GetParseError() << std::endl;
			document.SetObject(); //Reset to empty object
			return false;
		}

		//If loaded document is not an object, reset it
		if (!document.IsObject())
		{
			document.SetObject();
		}

		return true;
	}

	bool ShaderParameterLogger::logParameters(
		const std::string& imageName,
		const std::map<std::string, std::string>& tags,
		const std::map < std::string, float>& parameters)
	{
		rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

		//Create an entry for this image
		rapidjson::Value imageObject(rapidjson::kObjectType);

		//Add tags array
		rapidjson::Value tagsArray(rapidjson::kArrayType);
		for (const auto& tag : tags)
		{
			rapidjson::Value tagValue;
			tagValue.SetString(tag.second.c_str(), tag.second.length(), allocator);
			tagsArray.PushBack(tagValue, allocator);
		}
		imageObject.AddMember("tags", tagsArray, allocator);

		//Add parameters object
		rapidjson::Value paramsObject(rapidjson::kObjectType);
		for (const auto& param : parameters)
		{
			rapidjson::Value paramName;
			paramName.SetString(param.first.c_str(), param.first.length(), allocator);
			paramsObject.AddMember(paramName, param.second, allocator);
		}
		imageObject.AddMember("parameters", paramsObject, allocator);

		//Create image names as key
		rapidjson::Value imageNameKey;
		imageNameKey.SetString(imageName.c_str(), imageName.length(), allocator);

		//Add or replace the entry in the document
		if (document.HasMember(imageNameKey))
		{
			document[imageNameKey] = imageObject;
		}
		else 
		{
			document.AddMember(imageNameKey, imageObject, allocator);
		}

		return saveJsonFile();
	}

	bool ShaderParameterLogger::saveJsonFile()
	{
		//Open file for writing
		FILE* fp = fopen(jsonFilePath.c_str(), "wb");
		if (!fp)
		{
			std::cerr << "Error opening JSON file for writing: " << jsonFilePath << std::endl;
			return false;
		}

		std::vector<char> writeBuffer(65536);
		rapidjson::FileWriteStream os(fp, writeBuffer.data(), writeBuffer.size());


		//Use PrettyWriter for readable formatting
		rapidjson::PrettyWriter<rapidjson::FileWriteStream> writer(os);
		document.Accept(writer);

		fclose(fp);
		std::cout << "Parameters saved to JSON file: " << jsonFilePath << std::endl;
		return true;
	}
}

