#!/usr/bin/env python3
from util import post_request
import random


if __name__ == '__main__':
    payload = {
        "input": {
            "workflow": "txt2img",
            "payload": {
                "seed": random.randrange(1, 1000000),
                "steps": 20,
                "cfg_scale": 8,
                "sampler_name": "euler",
                "ckpt_name": "deliberate_v2.safetensors",
                "batch_size": 1,
                "width": 512,
                "height": 512,
                "prompt": "masterpiece best quality man wearing a hat",
                "negative_prompt": "bad hands"
            }
        },
        "webhookV2": "https://4c23-45-222-4-0.ngrok-free.app?token=helloworld"
    }

    post_request(payload)
