#!/usr/bin/env python3
from util import post_request
import random
import json


with open('9c803452-0ec2-46cd-b388-0129b8b0983c.json', 'r') as payload_file:
    payload_json = payload_file.read()

if __name__ == '__main__':
    payload = {
        "input": {
            "workflow": "custom",
            "payload": json.loads(payload_json)
        }
    }

#    seed = random.randrange(1, 1000000)

 #   payload["input"]["payload"]["3"]["inputs"]["seed"] = seed
 #   payload["input"]["payload"]["53"]["inputs"]["seed"] = seed

    post_request(payload)
