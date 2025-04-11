# AniGAN

AniGAN is an image generation API built using the Deep Convolutional Generative Adversarial Network (DCGAN) architecture. This project is based on the official DCGAN research paper and is designed with an API-first approach for simplicity and flexibility.

![image](https://github.com/user-attachments/assets/2ba654ef-baed-4d87-b23c-f676b4125b16)
\- Original


![image](https://github.com/user-attachments/assets/8362ccff-c146-4d56-9c54-fccf540cabc4)
\- AI-Generated

## Features

- DCGAN-based portrait image generation
- FastAPI-powered REST API
- Custom Generator and Discriminator implemented using PyTorch
- Statically served generated images
- Clean, backend-only implementation with no frontend

## Tech Stack

- Python
- PyTorch
- FastAPI
- TorchVision

## API Usage

### `GET /generate`

Generates a new portrait image and returns the URL for access.

#### Example Response

```json
{
  "url": "/generated/03fadc254f4b4760a8e7c681f0d51101.png"
}
```

You can generate images using:
`https://anigan-production.up.railway.app/generate`

Then, access them by:
`https://anigan-production.up.railway.app/generated/<image-name>.png`

## Local Development

To run AniGAN locally on your machine, follow the steps below:

### Prerequisites
- Python 3.10+
- `torch`, `torchvision`, and `fastapi`
- `uvicorn` (for running the server)

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/Aqua-16/AniGAN.git
cd AniGAN
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Model Weights**
Place your trained `generator_epoch_20.pth` and `discriminator_epoch_20.pth` files in the `checkpoints/` directory.

5. **Run the FastAPI server**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Your API will now be running on `http://localhost:8000`

You can test it by visiting: `http://localhost:8000/generate`


## References

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

---
Thank you for checking out my project!
