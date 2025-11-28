import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_runpod_logger():
    """Mock Runpod logger."""
    with patch('handler.RunPodLogger') as mock_logger:
        mock_instance = MagicMock()
        mock_logger.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_event():
    """Sample Runpod event for testing."""
    return {
        'id': 'test-job-123',
        'input': {
            'workflow': 'custom',
            'payload': {
                '3': {
                    'class_type': 'KSampler',
                    'inputs': {
                        'seed': 12345,
                        'steps': 20,
                        'cfg': 7.5,
                        'sampler_name': 'euler'
                    }
                },
                '9': {
                    'class_type': 'SaveImage',
                    'inputs': {
                        'filename_prefix': 'test'
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_txt2img_event():
    """Sample txt2img event for testing."""
    return {
        'id': 'test-job-456',
        'input': {
            'workflow': 'txt2img',
            'payload': {
                'seed': 12345,
                'steps': 20,
                'cfg_scale': 7.5,
                'sampler_name': 'euler',
                'ckpt_name': 'model.safetensors',
                'batch_size': 1,
                'width': 512,
                'height': 512,
                'prompt': 'a beautiful landscape',
                'negative_prompt': 'ugly, blurry'
            }
        }
    }


@pytest.fixture
def mock_comfyui_success_response():
    """Mock successful ComfyUI queue response."""
    return {
        'prompt_id': 'test-prompt-123'
    }


@pytest.fixture
def mock_comfyui_history_success():
    """Mock successful ComfyUI history response."""
    return {
        'test-prompt-123': {
            'status': {
                'status_str': 'success',
                'completed': True,
                'messages': []
            },
            'outputs': {
                '9': {
                    'images': [
                        {
                            'filename': 'test_00001_.png',
                            'type': 'output'
                        }
                    ]
                }
            }
        }
    }


@pytest.fixture
def mock_comfyui_history_error():
    """Mock failed ComfyUI history response."""
    return {
        'test-prompt-123': {
            'status': {
                'status_str': 'error',
                'completed': False,
                'messages': [
                    ['execution_error', {
                        'node_type': 'TestNode',
                        'exception_message': 'Test error message'
                    }]
                ]
            },
            'outputs': {}
        }
    }
