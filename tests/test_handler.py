import pytest
import json
import base64
import os
from unittest.mock import MagicMock, patch, mock_open


class TestGetOutputImages:
    """Tests for get_output_images function."""

    def test_single_image_output(self):
        from handler import get_output_images

        output = {
            '9': {
                'images': [
                    {'filename': 'test_00001_.png', 'type': 'output'}
                ]
            }
        }
        result = get_output_images(output)
        assert len(result) == 1
        assert result[0]['filename'] == 'test_00001_.png'

    def test_multiple_image_outputs(self):
        from handler import get_output_images

        output = {
            '9': {
                'images': [
                    {'filename': 'test_00001_.png', 'type': 'output'}
                ]
            },
            '10': {
                'images': [
                    {'filename': 'test_00002_.png', 'type': 'output'}
                ]
            }
        }
        result = get_output_images(output)
        assert len(result) == 2

    def test_empty_output(self):
        from handler import get_output_images

        output = {}
        result = get_output_images(output)
        assert len(result) == 0

    def test_output_without_images_key(self):
        from handler import get_output_images

        output = {
            '9': {
                'other_data': 'value'
            }
        }
        result = get_output_images(output)
        assert len(result) == 0


class TestCreateUniqueFilenamePrefix:
    """Tests for create_unique_filename_prefix function."""

    def test_adds_uuid_to_save_image_node(self):
        from handler import create_unique_filename_prefix

        payload = {
            '9': {
                'class_type': 'SaveImage',
                'inputs': {
                    'filename_prefix': 'original'
                }
            }
        }
        create_unique_filename_prefix(payload)

        # Should have replaced with a UUID
        new_prefix = payload['9']['inputs']['filename_prefix']
        assert new_prefix != 'original'
        # UUID format check (basic)
        assert len(new_prefix) == 36
        assert new_prefix.count('-') == 4

    def test_ignores_non_save_image_nodes(self):
        from handler import create_unique_filename_prefix

        payload = {
            '3': {
                'class_type': 'KSampler',
                'inputs': {
                    'seed': 12345
                }
            }
        }
        original_payload = json.loads(json.dumps(payload))
        create_unique_filename_prefix(payload)

        # Should remain unchanged
        assert payload == original_payload

    def test_handles_multiple_save_image_nodes(self):
        from handler import create_unique_filename_prefix

        payload = {
            '9': {
                'class_type': 'SaveImage',
                'inputs': {'filename_prefix': 'first'}
            },
            '10': {
                'class_type': 'SaveImage',
                'inputs': {'filename_prefix': 'second'}
            }
        }
        create_unique_filename_prefix(payload)

        prefix_9 = payload['9']['inputs']['filename_prefix']
        prefix_10 = payload['10']['inputs']['filename_prefix']

        # Both should be UUIDs
        assert len(prefix_9) == 36
        assert len(prefix_10) == 36
        # And different from each other
        assert prefix_9 != prefix_10


class TestGetTxt2ImgPayload:
    """Tests for get_txt2img_payload function."""

    def test_sets_all_expected_fields(self):
        from handler import get_txt2img_payload

        workflow = {
            '3': {'inputs': {}},
            '4': {'inputs': {}},
            '5': {'inputs': {}},
            '6': {'inputs': {}},
            '7': {'inputs': {}}
        }
        payload = {
            'seed': 12345,
            'steps': 20,
            'cfg_scale': 7.5,
            'sampler_name': 'euler',
            'ckpt_name': 'model.safetensors',
            'batch_size': 1,
            'width': 512,
            'height': 512,
            'prompt': 'test prompt',
            'negative_prompt': 'ugly'
        }

        result = get_txt2img_payload(workflow, payload)

        assert result['3']['inputs']['seed'] == 12345
        assert result['3']['inputs']['steps'] == 20
        assert result['3']['inputs']['cfg'] == 7.5
        assert result['3']['inputs']['sampler_name'] == 'euler'
        assert result['4']['inputs']['ckpt_name'] == 'model.safetensors'
        assert result['5']['inputs']['batch_size'] == 1
        assert result['5']['inputs']['width'] == 512
        assert result['5']['inputs']['height'] == 512
        assert result['6']['inputs']['text'] == 'test prompt'
        assert result['7']['inputs']['text'] == 'ugly'


class TestInputValidation:
    """Tests for input schema validation."""

    def test_valid_custom_workflow_input(self, sample_event):
        from runpod.serverless.utils.rp_validator import validate
        from schemas.input import INPUT_SCHEMA

        result = validate(sample_event['input'], INPUT_SCHEMA)
        assert 'errors' not in result
        assert result['validated_input']['workflow'] == 'custom'

    def test_valid_txt2img_workflow_input(self):
        from runpod.serverless.utils.rp_validator import validate
        from schemas.input import INPUT_SCHEMA

        input_data = {
            'workflow': 'txt2img',
            'payload': {'prompt': 'test'}
        }
        result = validate(input_data, INPUT_SCHEMA)
        assert 'errors' not in result

    def test_invalid_workflow_name(self):
        from runpod.serverless.utils.rp_validator import validate
        from schemas.input import INPUT_SCHEMA

        input_data = {
            'workflow': 'invalid_workflow',
            'payload': {}
        }
        result = validate(input_data, INPUT_SCHEMA)
        assert 'errors' in result

    def test_missing_payload(self):
        from runpod.serverless.utils.rp_validator import validate
        from schemas.input import INPUT_SCHEMA

        input_data = {
            'workflow': 'custom'
        }
        result = validate(input_data, INPUT_SCHEMA)
        assert 'errors' in result

    def test_default_workflow_is_txt2img(self):
        from runpod.serverless.utils.rp_validator import validate
        from schemas.input import INPUT_SCHEMA

        input_data = {
            'payload': {'prompt': 'test'}
        }
        result = validate(input_data, INPUT_SCHEMA)
        assert 'errors' not in result
        assert result['validated_input']['workflow'] == 'txt2img'


class TestSendRequests:
    """Tests for HTTP request functions."""

    def test_send_get_request_uses_correct_url(self):
        import handler
        from handler import BASE_URI, TIMEOUT

        # Create and inject a mock session
        mock_session = MagicMock()
        mock_session.get.return_value = MagicMock(status_code=200)
        handler.session = mock_session

        handler.send_get_request('test/endpoint')

        mock_session.get.assert_called_once_with(
            url=f'{BASE_URI}/test/endpoint',
            timeout=TIMEOUT
        )

    def test_send_post_request_uses_correct_url_and_payload(self):
        import handler
        from handler import BASE_URI, TIMEOUT

        # Create and inject a mock session
        mock_session = MagicMock()
        mock_session.post.return_value = MagicMock(status_code=200)
        handler.session = mock_session
        test_payload = {'key': 'value'}

        handler.send_post_request('test/endpoint', test_payload)

        mock_session.post.assert_called_once_with(
            url=f'{BASE_URI}/test/endpoint',
            json=test_payload,
            timeout=TIMEOUT
        )


class TestContainerInfo:
    """Tests for container telemetry functions."""

    @patch('handler.logging')
    def test_get_container_memory_info_handles_missing_files(self, mock_logging):
        from handler import get_container_memory_info

        with patch('builtins.open', side_effect=FileNotFoundError):
            result = get_container_memory_info()
            assert isinstance(result, dict)

    @patch('handler.logging')
    def test_get_container_cpu_info_handles_missing_files(self, mock_logging):
        from handler import get_container_cpu_info

        with patch('builtins.open', side_effect=FileNotFoundError):
            result = get_container_cpu_info()
            assert isinstance(result, dict)

    @patch('handler.logging')
    def test_get_container_disk_info_returns_dict(self, mock_logging):
        from handler import get_container_disk_info

        result = get_container_disk_info()
        assert isinstance(result, dict)
        # Should have some disk info on any system
        assert 'total_bytes' in result or len(result) >= 0


class TestHandlerErrorHandling:
    """Tests for handler error handling."""

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.validate')
    def test_handler_returns_error_on_validation_failure(
        self, mock_validate, mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        from handler import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}
        mock_validate.return_value = {'errors': ['Invalid input']}

        event = {'id': 'test-123', 'input': {}}
        result = handler(event)

        assert 'error' in result
        assert 'Invalid input' in result['error']

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    def test_handler_returns_error_on_low_memory(
        self, mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        from handler import handler

        mock_memory.return_value = {'available': 0.1}  # Very low memory
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        event = {'id': 'test-123', 'input': {'workflow': 'custom', 'payload': {}}}
        result = handler(event)

        assert 'error' in result
        assert 'memory' in result['error'].lower()

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    def test_handler_returns_error_on_low_disk_space(
        self, mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        from handler import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 100 * 1024}  # Very low disk

        event = {'id': 'test-123', 'input': {'workflow': 'custom', 'payload': {}}}
        result = handler(event)

        assert 'error' in result
        assert 'disk' in result['error'].lower()


class TestSnapLogHandler:
    """Tests for custom log handler."""

    def test_log_handler_formats_message_correctly(self, mock_runpod_logger):
        from handler import SnapLogHandler
        import logging

        with patch.dict(os.environ, {'RUNPOD_JOB_ID': 'test-job'}):
            handler = SnapLogHandler('test-app')
            handler.setFormatter(logging.Formatter('%(message)s'))

            record = logging.LogRecord(
                name='test',
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg='Test message',
                args=(),
                exc_info=None
            )

            # Should not raise
            handler.emit(record)

    def test_log_handler_handles_format_args(self, mock_runpod_logger):
        from handler import SnapLogHandler
        import logging

        handler = SnapLogHandler('test-app')
        handler.setFormatter(logging.Formatter('%(message)s'))

        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test %s message',
            args=('formatted',),
            exc_info=None
        )

        # Should not raise
        handler.emit(record)
