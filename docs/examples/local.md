# Local Examples

## Create and activate Python venv

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Remove credentials from .env 

If you have added your `RUNPOD_API_KEY` and
`RUNPOD_ENDPOINT_ID` to the `.env` file within
this directory, you should first comment them
out before attempting to test locally.  If
the .env file exists and the values are provided,
the examples will attempt to send the requests to
your Runpod endpoint instead of running locally.

## Run example scripts

Once the venv is created and activated, the requirements
installed, and the credentials removed from the .env
file, you can change directory to the `examples` directory
and run a script, for example:

```bash
cd examples
python txt2img.py
```

You obviously need to edit the payload within the
script to achieve the desired results.