from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from dcgan import DCGAN
from generator import Generator
from discriminator import Discriminator
from torchvision.utils import save_image
import uuid
import os

os.makedirs("generated", exist_ok=True)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/generated", StaticFiles(directory="generated"), name="generated")
G = Generator().to('cpu')
G.load_state_dict(torch.load(r'C:\Users\Mypc\OneDrive\Desktop\AniGAN\codebase\checkpoints\generator_epoch_20.pth', map_location='cpu'))
D = Discriminator().to('cpu')
D.load_state_dict(torch.load(r'C:\Users\Mypc\OneDrive\Desktop\AniGAN\codebase\checkpoints\discriminator_epoch_20.pth', map_location='cpu'))
model = DCGAN(G, D)



@app.get('/generate')
def generate():
    img = model.create()
    img_path = f"generated/{uuid.uuid4().hex}.png"
    os.makedirs("generated", exist_ok=True)
    save_image(img, img_path, normalize=True)
    return JSONResponse(content={"url": f"/{img_path}"})