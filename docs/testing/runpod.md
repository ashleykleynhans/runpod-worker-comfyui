# Runpod Testing

## Create and activate Python venv

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Add Runpod credentials to .env

Copy the `.env.example` file to '.env' as
follows:

```bash
cd examples
cp .env.example .env
```

Edit the .env file and add your RunPod API key to
`RUNPOD_API_KEY` and your endpoint ID to
`RUNPOD_ENDPOINT_ID`.  Without these credentials,
the tests will attempt to run locally instead of
on RunPod.

## Run test scripts

Once the venv is created and activated, the requirements
installed, and the credentials added to the .env
file, you can run a script, for example:

```bash
python txt2img.py
```

You obviously need to edit the payload within the
script to achieve the desired results.