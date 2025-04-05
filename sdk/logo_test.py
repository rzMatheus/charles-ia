from agents import Agent
from pydantic import BaseModel
import torch
from diffusers import FluxPipeline

# Define the LogoFeatures model
class LogoFeatures(BaseModel):
    briefing: str
    color: str
    is_lettered: bool

# Initialize the Logo Agent
logo_agent = Agent(
    name="Logo",
    instructions=(
        "Você criará um prompt ideal para gerar uma logo minimalista que remeta as informações "
        "da empresa (fornecidas por outro agente, o 'briefing_agent', que lhe retornará informações como a Identidade e propósito da empresa, História, Promessa e Motivação pessoal) "
        "e às características visuais mais adoradas pelo cliente (cor e se possui escrita ou não)"
    ),
)

# Initialize the pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Save VRAM if needed

# Function to create the logo prompt
def create_logo_prompt(briefing_info, logo_features: LogoFeatures):
    prompt = (
        f"Create a minimalistic logo reflecting {briefing_info['briefing']['identidade_proposito']}, "
        f"with a color scheme of {logo_features.color} and "
        f"{'using letters' if logo_features.is_lettered else 'not using letters'}, "
        f"that also captures the company history ({briefing_info['briefing']['historia']}), "
        f"promise ({briefing_info['briefing']['promessa']}), "
        f"and main motivation ({briefing_info['briefing']['motivacao']})."
    )
    return prompt

# Example input data
briefing_info = {
    'briefing': {
        'identidade_proposito': 'innovation and reliability',
        'historia': 'founded in 1990 as a tech startup',
        'promessa': 'committed to excellence',
        'motivacao': 'driving technological progress'
    }
}

# Simulating receiving logo features JSON and parsing it
logo_features_json = '{"briefing": "example_briefing", "color": "blue", "is_lettered": true}'
logo_features = LogoFeatures.parse_raw(logo_features_json)

# Generate the prompt using logo_agent
generated_prompt = create_logo_prompt(briefing_info, logo_features)

# Use FLUX Pipeline to generate the image
image = pipe(
    generated_prompt,
    height=128,
    width=128,
    guidance_scale=3.5,
    num_inference_steps=20,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

# Save the generated image
image.save("flux-dev.png")