
# 🧠 GENAI-EDIT-API

A versioned FastAPI-based service for image-to-image generation and editing using Stable Diffusion pipelines (img2img, instruct-pix2pix, and ControlNet).

---

## ✨ Features

- 🔁 **Level 1** – Image similarity generation (Stable Diffusion v1.4)
- ✏️ **Level 2** – Prompt-based image editing (InstructPix2Pix)
- 🖼️ **Level 3** – Dual-image guided generation (ControlNet with Canny + MLSD)

---

## 📦 Project Structure

```
app/
├── api/
│   ├── v1/ v2/ v3/           # Versioned endpoint modules
│   └── __init__.py
├── core/                     # Configs, middleware, exception handling
├── services/                 # Pipeline loaders
├── utils/                    # Image + prompt processing helpers
├── main.py                   # FastAPI app entry
├── plugin.py                 # Router registration
tests/
├── test_api.py               # Pytest-based test suite
Dockerfile
requirements.txt
README.md
```

---

## ✅ Prerequisites

1. **Docker Installation**  
   Follow [Docker’s official Ubuntu install guide](https://docs.docker.com/desktop/install/ubuntu/).

2. **NVIDIA GPU + Drivers**  
   Ensure NVIDIA GPU with proper driver support is installed.

3. **NVIDIA Container Toolkit**  
   Required for GPU access in containers. Install it:  
   👉 [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## 🚀 Run Locally with Docker

```bash
git clone  https://github.com/SerdarHelli/genai-edit-api

cd genai-edit-api

# Build Docker image
docker build -t genai-edit-api:test .

# Run with GPU support
docker run --rm -it \
  --gpus all \
  -p 9000:9000 \
  genai-edit-api:test
```

---

## 🔌 API Endpoints

Visit: [http://localhost:9000/docs](http://localhost:9000/docs) to view Swagger UI

### Level 1 – Image Similarity Generation
```
POST /api/v1/edit/level1
Form Data:
- image: UploadFile (JPEG)
- similarity_level: float (default: 0.8)
- prompt: Optional[str]
```

### Level 2 – Prompt-Based Editing
```
POST /api/v2/edit/level2
Form Data:
- image: UploadFile (JPEG)
- similarity_level: float (default: 0.8)
- prompt: str (e.g. "add a scratch")
```

### Level 3 – Dual-Image Guided Editing
```
POST /api/v3/edit/level3
Form Data:
- baseline: UploadFile (JPEG)
- annotated: UploadFile (JPEG)
- similarity_level: float (default: 0.8)
- prompt: Optional[str]
```

---

## 🧪 Run Tests

```
pytest tests/test_api.py
```

---

## 🛠 Models Used

| Level | Model | Pipeline |
|-------|-------|----------|
| 1     | `CompVis/stable-diffusion-v1-4` | `StableDiffusionImg2ImgPipeline` |
| 2     | `timbrooks/instruct-pix2pix`    | `StableDiffusionInstructPix2PixPipeline` |
| 3     | `runwayml/stable-diffusion-v1-5` + ControlNets | `StableDiffusionControlNetImg2ImgPipeline` |
|       | ControlNets: `lllyasviel/sd-controlnet-canny`, `lllyasviel/sd-controlnet-mlsd` | |

---

## 🤝 License

MIT License. © 2025 Serdar Helli
