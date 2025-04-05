from agents import Agent
from pydantic import BaseModel

class LogoFeatures(BaseModel):
    briefing:str
    color: str
    is_lettered: bool


# briefing_agent = Agent(
#     name="Briefing",
#     instructions="Você é um especialista de branding que conversará diretamente com o cliente e deve captar, através do briefing, os aspectos de Identidade e Propósito da Empresa, História, Promessa e Motivação principal da empresa. Pergunte ao usuário até que estes 4 aspectos sejam preenchidos. Uma vez preenchidos, guarde e envie em um JSON no seguinte formato {'briefing':{'identidade_propostito':'','historia':'','promessa':'motivacao':''}",
# )

logo_agent = Agent(
    name="Logo",
    instructions="Você criará um prompt ideal para gerar uma logo minimalista que remeta as informações da empresa (fornecidas por outro agente, o 'briefing_agent', que lhe retornará informações como a Identidade e propósito da empresa, História, Promessa e Motivação pessoal) e às características visuais mais adoradas pelo cliente (cor e se possui escrita ou não) ",
)

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

def create_logo_prompt(briefing_info, logo_features):
    # Here, the logo_agent processes the data
    prompt = (
        f"Create a minimalistic logo reflecting {briefing_info['briefing']['identidade_proposito']}, "
        f"with a color scheme of {logo_features['color']} and "
        f"{'using letters' if logo_features['is_lettered'] else 'not using letters'}, "
        f"that also captures the company history ({briefing_info['briefing']['historia']}), "
        f"promise ({briefing_info['briefing']['promessa']}), "
        f"and main motivation ({briefing_info['briefing']['motivacao']})."
    )
    return prompt

# Generate the prompt using logo_agent
generated_prompt = create_logo_prompt(briefing_info, logo_features)


prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=128,
    width=128,
    guidance_scale=3.5,
    num_inference_steps=20,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image.save("flux-dev.png")
