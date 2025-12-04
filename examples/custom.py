#!/usr/bin/env python3
from util import post_request
import random
import json


with open('comfyui-payload.json', 'r') as payload_file:
    payload_json = payload_file.read()

if __name__ == '__main__':
    payload = json.loads(payload_json)
    if not 'input' in payload:
        payload = {
            "input": payload
        }

    #print(json.dumps(payload, indent=2, default=str))

#    seed = random.randrange(1, 1000000)

 #   payload["input"]["payload"]["3"]["inputs"]["seed"] = seed
 #   payload["input"]["payload"]["53"]["inputs"]["seed"] = seed

    post_request(payload)
