# Deep Learning Foundations to Stable Diffusion - Lesson 9 Notes

## Course Overview

This lesson covers both practical usage of Stable Diffusion and the foundational concepts behind diffusion models. It is structured in two main parts:

1. **Practical Tutorial**: Using Hugging Face's Diffusers library to generate images with Stable Diffusion
2. **Theoretical Foundations**: Understanding the core concepts of how diffusion models work

### Key Concepts Covered:
- Stable Diffusion
- Hugging Face's Diffusers library
- Pre-trained pipelines
- Guidance scale
- Negative prompts
- Image-to-image generation
- Finite differencing
- Analytic derivatives
- Autoencoders
- Textual inversion
- Dreambooth
- Latent spaces
- U-Nets
- Text and image encoders
- Contrastive loss function
- CLIP text encoder
- Deep learning optimizers
- Perceptual loss

## Part 1: Using Stable Diffusion

### Getting Started with Diffusers

The lesson uses Hugging Face's Diffusers library, which provides pipelines for Stable Diffusion. This approach is similar to fastai's Learner concept, encapsulating models, processing, and inference in one object.

```python
# Basic usage
from diffusers import StableDiffusionPipeline

# Login to Hugging Face (first time only)
# huggingface-cli login

# Create pipeline from pre-trained model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Generate an image from a text prompt
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt).images[0]

# Display the image
image
```

### Key Parameters and Features

#### Guidance Scale

Guidance scale controls how closely the model follows the text prompt:
- Low values (1-3): More creative but less faithful to prompt
- Medium values (7-8): Good balance (default)
- High values (14+): Very literal but sometimes worse quality

```python
# Testing different guidance scales
prompts = ["an astronaut riding a horse"] * 4
images = pipe(prompts, guidance_scale=[1.0, 3.0, 7.5, 14.0]).images
```

#### Negative Prompts

You can exclude concepts from the generated image using negative prompts:

```python
# Generate a non-blue Labrador in Vermeer style
image = pipe(
    prompt="Labrador in the style of Vermeer",
    negative_prompt="blue",
).images[0]
```

#### Image-to-Image Generation

Start with an existing image and modify it according to a prompt:

```python
# Import image-to-image pipeline
from diffusers import StableDiffusionImg2ImgPipeline

# Create pipeline
i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Generate from initial image with varying strength
images = i2i_pipe(
    prompt="oil painting of a cat",
    image=init_image,
    strength=0.75  # How much to transform (0-1)
).images[0]
```

### Advanced Techniques

#### Textual Inversion

Teach the model a new concept with just a few example images:

1. Define a special token (e.g., "watercolor-portrait")
2. Add the token to the text model's vocabulary
3. Train the embedding with example images
4. Use the token in prompts

```python
# Using a trained textual inversion token
image = pipe("woman reading in the style of <watercolor-portrait>").images[0]
```

#### Dreambooth

Fine-tune a model to recognize a specific subject using a unique identifier:

1. Choose an uncommon token (e.g., "sks")
2. Fine-tune the model with images of your subject
3. Use the token in prompts

```python
# Using a Dreambooth fine-tuned model
image = db_pipe("painting of sks in the style of Paul Signac").images[0]
```

## Part 2: How Diffusion Models Work

### The Fundamental Concept

The instructor presents a novel way to understand diffusion models through a simple handwritten digit example. The core idea is:

1. Start with random noise
2. Gradually remove noise to produce a valid image
3. Guide this process using a neural network that predicts noise

### Building a Simple Diffusion Model

#### Step 1: Creating a Model to Identify Noise

To generate handwritten digits, we need a model that can:
1. Take a noisy image as input
2. Predict what noise was added to the original image
3. Allow us to subtract the predicted noise to recover the original

This model is trained by:
- Starting with clean digit images
- Adding varying amounts of random noise
- Training the model to predict the exact noise that was added

#### Step 2: Using the Model for Generation

Once trained, we can:
1. Start with pure random noise
2. Run the model to predict what noise is present
3. Subtract some of that noise
4. Repeat the process until we get a clean digit image

This iterative process is the core of all diffusion models.

### Components of Stable Diffusion

#### 1. The U-Net

The U-Net is the central component that:
- Takes noisy latents as input
- Predicts the noise in the input
- Allows us to subtract the noise

#### 2. The VAE (Variational Autoencoder)

The VAE compresses images to make processing more efficient:

- **Encoder**: Converts 512×512×3 images (786,432 values) into smaller "latents" (16,384 values)
- **Decoder**: Converts latents back into full images

This compression makes training and inference ~48x faster.

#### 3. The CLIP Text Encoder

To guide the generation toward specific images described by text:

1. CLIP consists of matched text and image encoders
2. Trained using contrastive learning to align text and image embeddings
3. The text encoder converts prompts into embeddings that guide the denoising process

#### 4. The Scheduler

The scheduler controls:
- How noise is added during training
- How the denoising steps are performed during generation
- The step size for each iteration

The instructor points out similarities between diffusion schedulers and deep learning optimizers, suggesting potential research directions.

## Conclusion

The lesson ends by highlighting that understanding these foundational concepts will allow us to:
1. Better use existing diffusion models
2. Develop novel approaches and improvements
3. Keep up with the rapidly evolving field

In future lessons, the course will build everything from scratch, starting from basic Python and working up to implementing all these components.
