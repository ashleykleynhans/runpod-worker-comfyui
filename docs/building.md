## Building the Docker image

You can either build this Docker image yourself, your alternatively,
you can use my pre-built image:

```
ashleykza/runpod-worker-comfyui:3.6.2
```

If you choose to build it yourself:

1. Sign up for a Docker hub account if you don't already have one.
2. Build the Docker image on your local machine and push to Docker hub:
```bash
# Clone the repo
git clone https://github.com/ashleykleynhans/runpod-worker-comfyui.git
cd runpod-worker-comfyui

# Build and push
docker build -t dockerhub-username/runpod-worker-comfyui:1.0.0 .
docker login
docker push dockerhub-username/runpod-worker-comfyui:1.0.0
```

If you're building on an M1 or M2 Mac, there will be an architecture
mismatch because they are `arm64`, but Runpod runs on `amd64`
architecture, so you will have to add the `--plaform` as follows:

```bash
docker buildx build --push -t dockerhub-username/runpod-worker-comfyui:1.0.0 . --platform linux/amd64
```
