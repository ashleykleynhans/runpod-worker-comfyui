variable "REGISTRY" {
    default = "docker.io"
}

variable "REGISTRY_USER" {
    default = "ashleykza"
}

variable "APP" {
    default = "runpod-worker-comfyui"
}

variable "RELEASE" {
    default = "4.0.1"
}

# CUDA 12.4 configuration
variable "CUDA_VERSION_124" {
    default = "12.4.1"
}

variable "TORCH_VERSION_124" {
    default = "2.6.0"
}

variable "XFORMERS_VERSION_124" {
    default = "0.0.29.post3"
}

# CUDA 12.8 configuration
variable "CUDA_VERSION_128" {
    default = "12.8.1"
}

variable "TORCH_VERSION_128" {
    default = "2.9.1"
}

variable "XFORMERS_VERSION_128" {
    default = "0.0.33"
}

group "default" {
    targets = ["cuda124", "cuda128"]
}

target "cuda124" {
    dockerfile = "Dockerfile"
    tags = ["${REGISTRY}/${REGISTRY_USER}/${APP}:${RELEASE}-cuda${CUDA_VERSION_124}"]
    args = {
        CUDA_VERSION = "${CUDA_VERSION_124}"
        TORCH_VERSION = "${TORCH_VERSION_124}"
        XFORMERS_VERSION = "${XFORMERS_VERSION_124}"
        CUDA_SHORT = "cu124"
    }
}

target "cuda128" {
    dockerfile = "Dockerfile"
    tags = ["${REGISTRY}/${REGISTRY_USER}/${APP}:${RELEASE}-cuda${CUDA_VERSION_128}"]
    args = {
        CUDA_VERSION = "${CUDA_VERSION_128}"
        TORCH_VERSION = "${TORCH_VERSION_128}"
        XFORMERS_VERSION = "${XFORMERS_VERSION_128}"
        CUDA_SHORT = "cu128"
    }
}
