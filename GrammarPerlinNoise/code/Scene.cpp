/*
* Codigo realizado por Hugo Montañés García.
*/

#include "Scene.hpp"

#include <iostream>
#include <cassert>


namespace space
{
	bool Scene::initFramebuffer(unsigned int width, unsigned int height)
	{
		// Reject invalid sizes
		if (width <= 0 || height <= 0) {
			std::cerr << "Error: framebuffer size must be >0\n";
			return false;
		}

		// Query hardware limits (optional sanity check)
		GLint maxTexSize = 0;
		glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexSize);
		if (width > maxTexSize || height > maxTexSize) {
			std::cerr << "Error: requested size exceeds GL_MAX_TEXTURE_SIZE ("
				<< maxTexSize << ")\n";
			return false;
		}

		fboWidth = width;
		fboHeight = height;

		// 1) Create FBO
		glGenFramebuffers(1, &framebufferObject);
		glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);

		// 2) Create and setup color texture
		glGenTextures(1, &frameTextureObject);
		glBindTexture(GL_TEXTURE_2D, frameTextureObject);

		// Use an explicit 8‐bit RGBA internal format
		glTexImage2D(GL_TEXTURE_2D, 0,
			GL_RGBA8,         // <— 8 bits per channel
			width, height,
			0,
			GL_RGBA,          // matches the internal format
			GL_UNSIGNED_BYTE,
			nullptr);

		// Filtering & clamping
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		// Attach it
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, frameTextureObject, 0);

		// 3) Create and attach depth buffer
		glGenRenderbuffers(1, &depthRenderBuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_RENDERBUFFER, depthRenderBuffer);

		// 4) Specify draw buffers
		GLenum drawBuf = GL_COLOR_ATTACHMENT0;
		glDrawBuffers(1, &drawBuf);

		// 6) Unbind everything
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		return true;
	}

	void Scene::deleteFramebuffer()
	{
		if (framebufferObject) {
			glDeleteFramebuffers(1, &framebufferObject);
			framebufferObject = 0;
		}
		if (frameTextureObject) {
			glDeleteTextures(1, &frameTextureObject);
			frameTextureObject = 0;
		}
		if (depthRenderBuffer) {
			glDeleteRenderbuffers(1, &depthRenderBuffer);
			depthRenderBuffer = 0;
		}
	}
	bool Scene::resizeFramebuffer(unsigned int width, unsigned int height)
	{
		// Delete old framebuffer
		deleteFramebuffer();
		// Create new framebuffer with new size
		return initFramebuffer(width, height);
	}
	void Scene::renderToFBO()
	{
		// Bind the FBO
		glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			std::cerr << "FBO incomplete before rendering to FBO: "<< fbStatusString(status) << "\n";
			// Don't return here - instead, recreate the FBO or handle the error differently

			// Try to recreate the framebuffer
			deleteFramebuffer();
			if (!initFramebuffer(fboWidth, fboHeight)) {
				std::cerr << "Failed to recreate framebuffer\n";
				return;
			}
			glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);
		}


		// Set the viewport to match the FBO size
		glViewport(0, 0, fboWidth, fboHeight);

		// Regular render code
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		if (!activeCamera) return;

		// Get camera matrices
		glm::mat4 view_matrix = activeCamera->getViewMatrix();
		glm::mat4 projection_matrix = activeCamera->getProjectionMatrix();

		// Send projection matrix to shader
		glUniformMatrix4fv(projection_matrix_id, 1, GL_FALSE, glm::value_ptr(projection_matrix));

		float currentTime = SDL_GetTicks() / 1000.0f;
		glUniform1f(time_id, currentTime);
		glUniform1f(noise_scale_id, 1.0f);

		glUniform1f(frequency_id, currentFrequency);
		glUniform1f(amplitude_id, currentAmplitude);
		glUniform1i(octaves_id, currentOctaves);

		// Render scene graph starting from root
		renderNode(root, view_matrix);

		// Make sure rendering is complete
		glFinish();

		GLenum err = glGetError();
		if (err != GL_NO_ERROR) {
			std::cerr << "OpenGL error after renderToFBO: 0x" << std::hex << err << std::dec << std::endl;
		}

	}
	bool Scene::saveFramebufferToFile(const std::string& filename,
		ScreenshotExporter::ImageFormat format)
	{
		if (!framebufferObject) {
			std::cerr << "Error: No FBO available for saving\n";
			return false;
		}

		assert(framebufferObject != 0 && "FrameBufferObject is zero! initFramebuffer must have failed.");

		// Bind on the unified target
		glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);
		glReadBuffer(GL_COLOR_ATTACHMENT0);

		GLint bound = 0;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &bound);
		std::cerr << "Currently bound FBO = " << bound << "\n";


		glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			std::cerr << "FBO not complete when saving: " << fbStatusString(status) << "\n";
			return false;
		}


		// Allocate buffer (check dims first)
		if (fboWidth <= 0 || fboHeight <= 0) {
			std::cerr << "Invalid FBO dimensions: "
				<< fboWidth << "x" << fboHeight << "\n";
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return false;
		}

		std::vector<unsigned char> pixels(fboWidth * fboHeight * 4);

		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0,
			fboWidth, fboHeight,
			GL_RGBA, GL_UNSIGNED_BYTE,
			pixels.data());

		// Check right away for GL errors
		GLenum err = glGetError();
		if (err != GL_NO_ERROR) {
			std::cerr << "OpenGL error in saveFramebufferToFile: 0x"
				<< std::hex << err << "\n";
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return false;
		}

		// Unbind
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// Finally write the file
		return screenshotExporter->saveImage(
			filename, fboWidth, fboHeight, pixels, format);
	}

	bool Scene::prepareFramebuffer() {
		// Check if FBO exists
		if (framebufferObject == 0) {
			return initFramebuffer(fboWidth, fboHeight);
		}

		// Bind and check status
		glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			std::cerr << "FBO incomplete, recreating: " << fbStatusString(status) << "\n";
			deleteFramebuffer();
			bool success = initFramebuffer(fboWidth, fboHeight);
			if (!success) {
				std::cerr << "Failed to recreate framebuffer\n";
				return false;
			}
		}

		return true;
	}

	Scene::Scene(unsigned width, unsigned height)
		: angle(0.0f), 
		framebufferObject(0),
		frameTextureObject(0),
		depthRenderBuffer(0),
		fboWidth(width),
		fboHeight(height)
	{
		glDisable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);
		glClearColor(0.2f, 0.2f, 0.2f, 1.f);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		window = SDL_GL_GetCurrentWindow();

		shader_program = std::make_unique<ShaderProgram>();

		screenshotExporter = std::make_unique<ScreenshotExporter>("../../../assets/generated_images");

		parameterLogger = std::make_unique<ShaderParameterLogger>("../../../assets/generated_images/tags.json");


		VertexShader vertex_shader;
		if (!vertex_shader.loadFromFile("../../../assets/shaders/vertexshaders/vertex_shader.glsl"))
		{
			throw std::runtime_error("Failed to load vertex shader.");
		}

		FragmentShader fragment_shader;
		if (!fragment_shader.loadFromFile("../../../assets/shaders/fragmentshaders/fragment_shader_voronoi.glsl"))
		{
			throw std::runtime_error("Failed to load fragment shader.");
		}

		shader_program->attachShader(vertex_shader);
		shader_program->attachShader(fragment_shader);

		if (!shader_program->link())
		{
			throw std::runtime_error("Failed to link shader program.");
		}

		shader_program->detachAndDeleteShaders({ vertex_shader, fragment_shader });

		shader_program->use();

		// Initialize the FBO
		if (!initFramebuffer(width, height)) {
			throw std::runtime_error("Failed to initialize FBO!");
		}

		glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		std::cerr << "Initial FBO status: " << fbStatusString(status)<<"\n";
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		std::cerr << "After initFramebuffer, FBO ID = "
			<< framebufferObject << "\n";

		// Get uniform locations
		model_view_matrix_id = glGetUniformLocation(shader_program->getProgramID(), "model_view_matrix");
		normal_matrix_id = glGetUniformLocation(shader_program->getProgramID(), "normal_matrix");
		projection_matrix_id = glGetUniformLocation(shader_program->getProgramID(), "projection_matrix");

		noise_scale_id = glGetUniformLocation(shader_program->getProgramID(), "noise_scale");
		time_id = glGetUniformLocation(shader_program->getProgramID(), "time");

		frequency_id = glGetUniformLocation(shader_program->getProgramID(), "frequency");
		amplitude_id = glGetUniformLocation(shader_program->getProgramID(), "amplitude");
		octaves_id = glGetUniformLocation(shader_program->getProgramID(), "octaves");
		
		root = std::make_shared<SceneNode>("root");
		//Root node

		//Setup camera
		activeCamera = std::make_shared<Camera>("main_camera");
		activeCamera->position = glm::vec3(0, 20, 50);
		activeCamera->rotation = glm::vec3(-0.4f, 0.0f, 0.0f);
		root->addChild(activeCamera);

		auto quadNode = std::make_shared<SceneNode>("screen_quad");
		quadNode->mesh = std::make_shared<ScreenQuad>();
		root->addChild(quadNode);


		GLenum error = glGetError();
		if (error != GL_NO_ERROR)
		{
			std::cerr << "OpenGL error in Scene constructor: " << error << std::endl;
		}
	}


	void Scene::update(float deltaTime)
	{

		//Get current keyboard state
		const Uint8* keyboardState = SDL_GetKeyboardState(nullptr);
		handleKeyboard(keyboardState);

	}

	void Scene::renderNode(const std::shared_ptr<SceneNode>& node, const glm::mat4& viewMatrix)
	{
		if (node->mesh)
		{
			shader_program->use();

			// Special case for screen quad
			if (node->name == "screen_quad") {
				// Use identity matrices for the screen quad
				glm::mat4 identity(1.0f);
				glUniformMatrix4fv(model_view_matrix_id, 1, GL_FALSE, glm::value_ptr(identity));
				glUniformMatrix4fv(normal_matrix_id, 1, GL_FALSE, glm::value_ptr(identity));
				glUniformMatrix4fv(projection_matrix_id, 1, GL_FALSE, glm::value_ptr(identity));
			}
			else {
				// Regular model-view-projection for other objects
				glm::mat4 model_matrix = node->getWorldTransform();
				glm::mat4 model_view_matrix = viewMatrix * model_matrix;
				glm::mat4 normal_matrix = glm::transpose(glm::inverse(model_view_matrix));

				glUniformMatrix4fv(model_view_matrix_id, 1, GL_FALSE, glm::value_ptr(model_view_matrix));
				glUniformMatrix4fv(normal_matrix_id, 1, GL_FALSE, glm::value_ptr(normal_matrix));
			}

			// Render the mesh
			node->mesh->render();
		}

		// Recursively render children
		for (const auto& child : node->children) {
			renderNode(child, viewMatrix);
		}
	}

	void Scene::render()
	{

		shader_program->use();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		if (!activeCamera) return;

		// Get camera matrices
		glm::mat4 view_matrix = activeCamera->getViewMatrix();
		glm::mat4 projection_matrix = activeCamera->getProjectionMatrix();

		// Send projection matrix to shader
		glUniformMatrix4fv(projection_matrix_id, 1, GL_FALSE, glm::value_ptr(projection_matrix));

		float currentTime = SDL_GetTicks() / 1000.0f;
		glUniform1f(time_id, currentTime);
		glUniform1f(noise_scale_id, 1.0f); // Adjust this value to change noise scale

		glUniform1f(frequency_id, currentFrequency);
		glUniform1f(amplitude_id, currentAmplitude);
		glUniform1i(octaves_id, currentOctaves);

		// Render scene graph starting from root
		renderNode(root, view_matrix);

		GLenum error = glGetError(); 
		if (error != GL_NO_ERROR) 
		{ 
			std::cerr << "OpenGL error in render: " << error << std::endl; 
		}
	}

	void Scene::resize(unsigned width, unsigned height)
	{
		if (activeCamera) {
			activeCamera->aspect = float(width) / height;
		}
		glViewport(0, 0, width, height);

		// Resize the FBO as well
		resizeFramebuffer(width, height);

		GLenum error = glGetError();
		if (error != GL_NO_ERROR) {
			std::cerr << "OpenGL error in resize: " << error << std::endl;
		}
	}

	// Utility functions to manipulate scene
	std::shared_ptr<SceneNode> Scene::createNode(const std::string& name,
		std::shared_ptr<SceneNode> parent) {
		auto node = std::make_shared<SceneNode>(name);
		if (!parent) parent = root;
		parent->addChild(node);
		return node;
	}

	std::shared_ptr<SceneNode> Scene::findNode(const std::string& name,
		const std::shared_ptr<SceneNode>& startNode = nullptr) {
		auto node = startNode ? startNode : root;
		if (node->name == name) return node;

		for (const auto& child : node->children) {
			if (auto found = findNode(name, child)) {
				return found;
			}
		}
		return nullptr;
	}
	void Scene::handleKeyboard(const Uint8* keyboardState)
	{

		//Screenshot key - check if Z was just pressed
		static bool ZWasPressed = false;
		bool ZIsPressed = keyboardState[SDL_SCANCODE_Z];

		if (ZIsPressed && !ZWasPressed)
		{
			takeScreenshot(ScreenshotExporter::ImageFormat::PNG);
		}

		ZWasPressed = ZIsPressed;


		static bool EscWasPressed = false;
		bool EscIsPressed = keyboardState[SDL_SCANCODE_ESCAPE];

		if (EscIsPressed && !EscWasPressed)
		{
			exitRequested = true;
			SDL_Quit();
		}

		EscWasPressed = EscIsPressed;
	}

	bool Scene::takeScreenshot(ScreenshotExporter::ImageFormat format) {
		// Save current framebuffer binding
		GLint previousFBO;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFBO);

		// Bind and render to our FBO
		glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);

		// Render scene to FBO
		renderToFBO();

		// Check FBO status
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			std::cerr << "FBO not complete when taking screenshot: " << fbStatusString(status) << "\n";
			glBindFramebuffer(GL_FRAMEBUFFER, previousFBO); // Restore previous binding
			return false;
		}

		// Generate filename
		std::string filename = screenshotExporter->getOutputPath() + "image_" +
			std::to_string(screenshotExporter->getLastImageCounter() + 1);

		// Add file extension
		switch (format) {
		case ScreenshotExporter::ImageFormat::PNG:
			filename += ".png";
			break;
		case ScreenshotExporter::ImageFormat::JPG:
			filename += ".jpg";
			break;
		}

		// Save the FBO content to file
		bool result = saveFramebufferToFile(filename, format);

		// Restore previous framebuffer binding
		glBindFramebuffer(GL_FRAMEBUFFER, previousFBO);

		if (result) {
			// Increment counter and log parameters
			screenshotExporter->incrementCounter();

			std::map<std::string, float> parameters = {
				{"frequency", currentFrequency},
				{"amplitude", currentAmplitude},
				{"octaves", static_cast<float>(currentOctaves)}
			};

			std::map<std::string, std::string> tags = {
				{"0", "voronoi"}
			};

			// Extract just the filename without path
			size_t lastSlash = filename.find_last_of("/\\");
			std::string justFilename = filename.substr(lastSlash + 1);

			parameterLogger->logParameters(justFilename, tags, parameters);
		}

		return result;
	}

	bool Scene::checkFramebufferStatus()
	{
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE)
		{
			std::cerr << "Framebuffer incomplete. Status: " << status << std::endl;
			return false;
		}
		return true;
	}

	Scene::~Scene() {
		deleteFramebuffer();
	}

	

}

