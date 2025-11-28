# Unit Tests

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

## Running Tests

Run all tests with verbose output and coverage:

```bash
pytest -v
```

Coverage is automatically included and will display after the test results.

## Running Specific Tests

Run a specific test file:

```bash
pytest tests/test_handler.py -v
```

Run a specific test class:

```bash
pytest tests/test_handler.py::TestGetOutputImages -v
```

Run a specific test:

```bash
pytest tests/test_handler.py::TestGetOutputImages::test_single_image_output -v
```

## Coverage Reports

Generate an HTML coverage report:

```bash
pytest --cov-report=html
open htmlcov/index.html
```

Generate an XML coverage report (for CI):

```bash
pytest --cov-report=xml
```

## Test Markers

Skip slow tests:

```bash
pytest -v -m "not slow"
```

Run only integration tests:

```bash
pytest -v -m integration
```
