# Deep Learning Foundations to Stable Diffusion - Lesson 9 Notes

## Course Information

- This is Lesson 9, the first lesson of Part 2 of the "Practical Deep Learning for Coders" series
- It's recommended to have completed Part 1 or have equivalent knowledge:
  - Understand basic deep learning concepts
  - Be able to write a basic SGD loop in Python
  - Know how to use PyTorch or TensorFlow
  - Understand what embeddings are and how to create them
- Expect to spend about 10 hours of work on each video

## Lesson Overview

This lesson is divided into two main parts:
1. **Tutorial on using Diffusers library for image generation**
2. **Understanding the key concepts behind Stable Diffusion**

## Part 1: Getting Started with Stable Diffusion

### Using the Diffusers Library

- Hugging Face's Diffusers library is currently recommended for working with Stable Diffusion
- Need to log in to Hugging Face to access models
- Using pipelines (similar to fastai Learners):
  ```python
  from diffusers import StableDiffusionPipeline
  
  pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
  prompt = "a photograph of an astronaut riding a horse"
  image = pipe(prompt).images[0]
  ```

### Key Parameters and Features

#### Guidance Scale
- Controls how much to focus on the prompt vs. general image creation
- Default is around 7.5
- Lower values (e.g., 1) produce more creative but less prompt-focused images
- Higher values (e.g., 14) produce images that closely follow the prompt but may look less natural

#### Negative Prompts
- Allows specifying what you don't want in the image
- Example: "Labrador in the style of Vermeer" with negative prompt "blue" produces a non-blue Labrador

#### Image-to-Image Pipeline
- Start with an existing image rather than random noise
- Useful for guiding the composition or improving a sketch
- Parameter "strength" controls how much to preserve of the original image

#### Iterative Generation
- Can use output images as inputs for further refinement
- Example: Generate an initial image, then use it as input with a new prompt like "oil painting by Van Gogh"

### Fine-tuning Approaches

#### Standard Fine-tuning
- Train the model on new images with captions
- Example: Training on Pokémon images with auto-generated captions

#### Textual Inversion
- Fine-tune just a single embedding rather than the whole model
- Create a new token that represents a specific visual concept
- Requires just a few example images (e.g., 4 watercolor portraits)
- Much faster than full model fine-tuning

#### DreamBooth
- Takes an unusual token (e.g., "sks") and associates it with specific subject images
- Can generate new images of the subject in different styles
- Example: Training on photos of a person to generate paintings of them

### Resources

- **Repo**: "diffusion-nbs" contains notebooks to play with
- **Tools list**: Jonathan Whitaker's "suggested_tools.md"
- **Prompts database**: Lexica.art for exploring prompts and their outputs
- **Computing options**: Colab, Paperspace Gradient, Lambda Labs, Jarvis Labs
  - GPU requirements and pricing rapidly changing due to Stable Diffusion popularity

## Part 2: Understanding How Stable Diffusion Works

Jeremy introduces a new conceptual framework for understanding diffusion models:

### 1. Noise Prediction Concept

Starting with a simplified example of generating handwritten digits:

- Imagine a function that can tell the probability that an image is a handwritten digit
- Could use this function to gradually transform random noise into a digit:
  - For each pixel, determine if making it lighter/darker would increase the probability
  - This gives us the gradient: how to change each pixel to make it more digit-like
  - Apply these changes iteratively to transform noise into a digit

### 2. Neural Network for Noise Prediction

- Instead of using finite differencing (checking each pixel individually)
- Train a Neural Net to directly predict the noise in an image
- Training process:
  1. Take real digits and add random amounts of noise
  2. Train the network to predict what noise was added
  3. Once trained, feed in pure noise
  4. Network predicts what part is noise, subtract it
  5. Repeat to gradually denoise

### 3. Components of Stable Diffusion

#### U-Net
- The core component that predicts noise
- Input: Somewhat noisy image (or latents)
- Output: The noise itself
- Subtracting this predicted noise from the input creates a less noisy image

#### VAE (Variational Autoencoder)
- Used for compression to make the process more efficient
- Consists of:
  - Encoder: Compresses images into "latents" (smaller representation)
  - Decoder: Expands latents back into full images
- Working with latents (e.g., 64×64×4 = 16,384 values) is much faster than full images (512×512×3 = 786,432 values)
- The U-Net operates on these latents instead of full images

#### CLIP Text Encoder
- Allows the model to be guided by text prompts
- Created through contrastive learning:
  - Two models trained together: text encoder and image encoder
  - Trained to make embeddings similar for matching text-image pairs
  - And dissimilar for non-matching pairs
- During generation, text embeddings guide the denoising process

#### Scheduler/Sampler
- Controls how noise is added during training and removed during generation
- Determines step size, noise schedule, etc.
- Similar to optimizers in regular deep learning

### The Generation Process

1. Start with random noise (in latent space)
2. Feed noise + text embedding through U-Net to predict what noise to remove
3. Subtract some of the predicted noise (not all at once)
4. Repeat steps 2-3 multiple times, gradually denoising
5. Convert final latents to image using VAE decoder

### Future Research Directions

- Treating diffusion as an optimization problem rather than a differential equation
- Experimenting with different loss functions (e.g., perceptual loss instead of MSE)
- Removing the reliance on "t" (timestep) parameter
- Using techniques from optimizers like momentum and Adam

## Course Plan Going Forward

- Next lesson: Looking inside the pipeline to understand the code
- Then: Building everything from the foundations using only the Python standard library
- Goal: Recreate diffusion models from scratch and explore new research directions

## Additional Resources

- Visit course.fast.ai for all materials
- Join forums.fast.ai for community discussions
- Check out additional videos from course contributors

## Key Contributors

- Jeremy Howard (fastai)
- Jonathan Whitaker (first detailed educational material on Stable Diffusion)
- Wasim (fastai contributor)
- Petro (Hugging Face, working on diffusers)
- Tanishq (Stability.AI, working on Stable Diffusion models)
