import pytest
import json
import base64
import os
import logging
import requests
from unittest.mock import MagicMock, patch, mock_open, call


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

        new_prefix = payload['9']['inputs']['filename_prefix']
        assert new_prefix != 'original'
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

        assert len(prefix_9) == 36
        assert len(prefix_10) == 36
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


class TestGetImg2ImgPayload:
    """Tests for get_img2img_payload function."""

    def test_sets_all_expected_fields(self):
        from handler import get_img2img_payload

        workflow = {
            '1': {'inputs': {}},
            '2': {'inputs': {}},
            '4': {'inputs': {}},
            '6': {'inputs': {}},
            '7': {'inputs': {}},
            '13': {'inputs': {}}
        }
        payload = {
            'seed': 12345,
            'steps': 20,
            'cfg_scale': 7.5,
            'sampler_name': 'euler',
            'scheduler': 'normal',
            'denoise': 0.75,
            'ckpt_name': 'model.safetensors',
            'width': 512,
            'height': 512,
            'prompt': 'test prompt',
            'negative_prompt': 'ugly'
        }

        result = get_img2img_payload(workflow, payload)

        assert result['13']['inputs']['seed'] == 12345
        assert result['13']['inputs']['steps'] == 20
        assert result['13']['inputs']['cfg'] == 7.5
        assert result['13']['inputs']['sampler_name'] == 'euler'
        assert result['13']['inputs']['scheduler'] == 'normal'
        assert result['13']['inputs']['denoise'] == 0.75
        assert result['1']['inputs']['ckpt_name'] == 'model.safetensors'
        assert result['2']['inputs']['width'] == 512
        assert result['2']['inputs']['height'] == 512
        assert result['6']['inputs']['text'] == 'test prompt'
        assert result['7']['inputs']['text'] == 'ugly'


class TestGetWorkflowPayload:
    """Tests for get_workflow_payload function."""

    def test_loads_txt2img_workflow(self):
        from handler import get_workflow_payload

        mock_workflow = {
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
            'prompt': 'test',
            'negative_prompt': 'ugly'
        }

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_workflow))):
            result = get_workflow_payload('txt2img', payload)

        assert result['3']['inputs']['seed'] == 12345
        assert result['6']['inputs']['text'] == 'test'

    def test_loads_custom_workflow_without_modification(self):
        from handler import get_workflow_payload

        mock_workflow = {'custom': 'workflow'}

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_workflow))):
            result = get_workflow_payload('other', {})

        assert result == mock_workflow


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


class TestWaitForService:
    """Tests for wait_for_service function."""

    @patch('handler.time.sleep')
    @patch('handler.requests.get')
    def test_returns_immediately_on_success(self, mock_get, mock_sleep):
        from handler import wait_for_service

        mock_get.return_value = MagicMock(status_code=200)

        wait_for_service('http://localhost:3000')

        mock_get.assert_called_once_with('http://localhost:3000')
        mock_sleep.assert_not_called()

    @patch('handler.logging')
    @patch('handler.time.sleep')
    @patch('handler.requests.get')
    def test_retries_on_request_exception(self, mock_get, mock_sleep, mock_logging):
        from handler import wait_for_service

        mock_get.side_effect = [
            requests.exceptions.ConnectionError(),
            MagicMock(status_code=200)
        ]

        wait_for_service('http://localhost:3000')

        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 1

    @patch('handler.logging')
    @patch('handler.time.sleep')
    @patch('handler.requests.get')
    def test_logs_every_15_retries(self, mock_get, mock_sleep, mock_logging):
        from handler import wait_for_service

        # Fail 15 times then succeed
        side_effects = [requests.exceptions.ConnectionError()] * 15 + [MagicMock(status_code=200)]
        mock_get.side_effect = side_effects

        wait_for_service('http://localhost:3000')

        assert mock_get.call_count == 16
        mock_logging.info.assert_called()

    @patch('handler.logging')
    @patch('handler.time.sleep')
    @patch('handler.requests.get')
    def test_logs_on_general_exception(self, mock_get, mock_sleep, mock_logging):
        from handler import wait_for_service

        mock_get.side_effect = [
            Exception('Some error'),
            MagicMock(status_code=200)
        ]

        wait_for_service('http://localhost:3000')

        mock_logging.error.assert_called()


class TestContainerMemoryInfo:
    """Tests for get_container_memory_info function."""

    @patch('handler.logging')
    def test_reads_proc_meminfo(self, mock_logging):
        from handler import get_container_memory_info

        meminfo_content = """MemTotal:       16384000 kB
MemFree:         8192000 kB
MemAvailable:   12288000 kB
"""
        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/meminfo':
                m = mock_open(read_data=meminfo_content)()
                m.readlines.return_value = meminfo_content.strip().split('\n')
                return m
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_memory_info()

        assert 'total' in result
        assert 'free' in result
        assert 'available' in result
        assert 'used' in result

    @patch('handler.logging')
    def test_reads_cgroups_v2(self, mock_logging):
        from handler import get_container_memory_info

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/meminfo':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/memory.max':
                return mock_open(read_data='8589934592')()
            elif path == '/sys/fs/cgroup/memory.current':
                return mock_open(read_data='4294967296')()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_memory_info()

        assert 'limit' in result
        assert 'used' in result
        assert 'available' in result

    @patch('handler.logging')
    def test_reads_cgroups_v1(self, mock_logging):
        from handler import get_container_memory_info

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/meminfo':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/memory.max':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/memory/memory.limit_in_bytes':
                return mock_open(read_data='8589934592')()
            elif path == '/sys/fs/cgroup/memory/memory.usage_in_bytes':
                return mock_open(read_data='4294967296')()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_memory_info()

        assert 'limit' in result
        assert 'used' in result

    @patch('handler.logging')
    def test_reads_cgroups_v1_alt_path(self, mock_logging):
        from handler import get_container_memory_info

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/meminfo':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/memory.max':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/memory/memory.limit_in_bytes':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/memory.limit_in_bytes':
                return mock_open(read_data='8589934592')()
            elif path == '/sys/fs/cgroup/memory.usage_in_bytes':
                return mock_open(read_data='4294967296')()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_memory_info()

        assert 'limit' in result
        assert 'used' in result

    @patch('handler.logging')
    def test_handles_unlimited_memory(self, mock_logging):
        from handler import get_container_memory_info

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/meminfo':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/memory.max':
                return mock_open(read_data='max')()
            elif path == '/sys/fs/cgroup/memory.current':
                return mock_open(read_data='4294967296')()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_memory_info()

        assert 'limit' not in result

    @patch('handler.logging')
    def test_handles_large_limit_cgroups_v1(self, mock_logging):
        from handler import get_container_memory_info

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/meminfo':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/memory.max':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/memory/memory.limit_in_bytes':
                return mock_open(read_data=str(2**63 + 1000))()
            elif path == '/sys/fs/cgroup/memory/memory.usage_in_bytes':
                return mock_open(read_data='4294967296')()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_memory_info()

        assert 'limit' not in result

    @patch('handler.logging')
    def test_logs_no_memory_info(self, mock_logging):
        from handler import get_container_memory_info

        with patch('builtins.open', side_effect=FileNotFoundError()):
            result = get_container_memory_info()

        assert result == {}

    @patch('handler.logging')
    def test_handles_exception(self, mock_logging):
        from handler import get_container_memory_info

        with patch('builtins.open', side_effect=Exception('Test error')):
            result = get_container_memory_info()

        assert result == {}
        mock_logging.error.assert_called()


class TestContainerCPUInfo:
    """Tests for get_container_cpu_info function."""

    @patch('handler.logging')
    def test_reads_proc_cpuinfo(self, mock_logging):
        from handler import get_container_cpu_info

        cpuinfo_lines = [
            "processor\t: 0\n",
            "model name\t: Intel\n",
            "processor\t: 1\n",
            "model name\t: Intel\n"
        ]

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/cpuinfo':
                m = MagicMock()
                m.__enter__ = MagicMock(return_value=iter(cpuinfo_lines))
                m.__exit__ = MagicMock(return_value=False)
                return m
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_cpu_info()

        assert result.get('available_cpus') == 2

    @patch('handler.logging')
    def test_reads_cgroups_v2_cpu(self, mock_logging):
        from handler import get_container_cpu_info

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/cpuinfo':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu.max':
                return mock_open(read_data='200000 100000')()
            elif path == '/sys/fs/cgroup/cpu.stat':
                return mock_open(read_data='usage_usec 123456789')()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_cpu_info()

        assert result.get('allocated_cpus') == 2.0
        assert 'usage_usec' in result

    @patch('handler.logging')
    def test_reads_cgroups_v1_cpu(self, mock_logging):
        from handler import get_container_cpu_info

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/cpuinfo':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu.max':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu/cpu.cfs_quota_us':
                return mock_open(read_data='200000')()
            elif path == '/sys/fs/cgroup/cpu/cpu.cfs_period_us':
                return mock_open(read_data='100000')()
            elif path == '/sys/fs/cgroup/cpu.stat':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu/cpuacct.usage':
                return mock_open(read_data='123456789000')()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_cpu_info()

        assert result.get('allocated_cpus') == 2.0

    @patch('handler.logging')
    def test_reads_cgroups_v1_alt_path(self, mock_logging):
        from handler import get_container_cpu_info

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/cpuinfo':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu.max':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu/cpu.cfs_quota_us':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu.cfs_quota_us':
                return mock_open(read_data='200000')()
            elif path == '/sys/fs/cgroup/cpu.cfs_period_us':
                return mock_open(read_data='100000')()
            elif path == '/sys/fs/cgroup/cpu.stat':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu/cpuacct.usage':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpuacct.usage':
                return mock_open(read_data='123456789000')()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_cpu_info()

        assert result.get('allocated_cpus') == 2.0

    @patch('handler.logging')
    def test_handles_max_cpu(self, mock_logging):
        from handler import get_container_cpu_info

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/cpuinfo':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu.max':
                return mock_open(read_data='max 100000')()
            elif path == '/sys/fs/cgroup/cpu.stat':
                raise FileNotFoundError()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_cpu_info()

        assert 'allocated_cpus' not in result

    @patch('handler.logging')
    def test_handles_negative_quota(self, mock_logging):
        from handler import get_container_cpu_info

        def mock_open_func(path, *args, **kwargs):
            if path == '/proc/cpuinfo':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu.max':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/cpu/cpu.cfs_quota_us':
                return mock_open(read_data='-1')()
            elif path == '/sys/fs/cgroup/cpu/cpu.cfs_period_us':
                return mock_open(read_data='100000')()
            elif path == '/sys/fs/cgroup/cpu.stat':
                raise FileNotFoundError()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            result = get_container_cpu_info()

        assert 'allocated_cpus' not in result

    @patch('handler.logging')
    def test_handles_exception(self, mock_logging):
        from handler import get_container_cpu_info

        with patch('builtins.open', side_effect=Exception('Test error')):
            result = get_container_cpu_info()

        assert result == {}
        mock_logging.error.assert_called()


class TestContainerDiskInfo:
    """Tests for get_container_disk_info function."""

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_gets_disk_usage_with_job_id(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)

        with patch('builtins.open', side_effect=FileNotFoundError()):
            with patch('os.statvfs') as mock_statvfs:
                mock_statvfs.return_value = MagicMock(
                    f_files=1000000,
                    f_ffree=500000
                )
                result = get_container_disk_info(job_id='test-job-123')

        assert result['total_bytes'] == 100 * 1024**3
        mock_logging.info.assert_called()

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_no_disk_info_with_job_id(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.side_effect = Exception('Disk error')

        with patch('builtins.open', side_effect=FileNotFoundError()):
            with patch('os.statvfs', side_effect=Exception('statvfs error')):
                result = get_container_disk_info(job_id='test-job-123')

        # Empty result but with job_id path coverage
        assert result == {} or 'total_bytes' not in result

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_disk_info_logs_io_bytes(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)

        io_content = "8:0 Read 1234567\n8:0 Write 7654321\n8:0 Total 123456789\n"

        def mock_open_func(path, *args, **kwargs):
            if path == '/sys/fs/cgroup/io.stat':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/blkio/blkio.throttle.io_service_bytes':
                m = MagicMock()
                m.__enter__ = MagicMock(return_value=iter(io_content.split('\n')))
                m.__exit__ = MagicMock(return_value=False)
                return m
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            with patch('os.statvfs') as mock_statvfs:
                mock_statvfs.return_value = MagicMock(f_files=1000000, f_ffree=500000)
                result = get_container_disk_info(job_id='test-job')

        assert result.get('io_bytes') == 123456789

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_gets_disk_usage(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)

        with patch('builtins.open', side_effect=FileNotFoundError()):
            with patch('os.statvfs') as mock_statvfs:
                mock_statvfs.return_value = MagicMock(
                    f_files=1000000,
                    f_ffree=500000
                )
                result = get_container_disk_info()

        assert result['total_bytes'] == 100 * 1024**3
        assert result['used_bytes'] == 50 * 1024**3
        assert result['free_bytes'] == 50 * 1024**3

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_reads_io_stats_v2(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)

        def mock_open_func(path, *args, **kwargs):
            if path == '/sys/fs/cgroup/io.stat':
                return mock_open(read_data='253:0 rbytes=123456 wbytes=654321')()
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            with patch('os.statvfs') as mock_statvfs:
                mock_statvfs.return_value = MagicMock(f_files=1000000, f_ffree=500000)
                result = get_container_disk_info()

        assert 'io_stats_raw' in result

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_reads_io_stats_v1(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)

        # Format: "device operation bytes" where we look for 'Total' and parts[2]
        io_content = "8:0 Read 1234567\n8:0 Write 7654321\n8:0 Total 123456789\n"

        def mock_open_func(path, *args, **kwargs):
            if path == '/sys/fs/cgroup/io.stat':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/blkio/blkio.throttle.io_service_bytes':
                m = MagicMock()
                m.__enter__ = MagicMock(return_value=iter(io_content.split('\n')))
                m.__exit__ = MagicMock(return_value=False)
                return m
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            with patch('os.statvfs') as mock_statvfs:
                mock_statvfs.return_value = MagicMock(f_files=1000000, f_ffree=500000)
                result = get_container_disk_info()

        assert result.get('io_bytes') == 123456789

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_reads_io_stats_v1_alt(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)

        # Format: "device operation bytes" where we look for 'Total' and parts[2]
        io_content = "8:0 Read 1234567\n8:0 Write 7654321\n8:0 Total 123456789\n"

        def mock_open_func(path, *args, **kwargs):
            if path == '/sys/fs/cgroup/io.stat':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/blkio/blkio.throttle.io_service_bytes':
                raise FileNotFoundError()
            elif path == '/sys/fs/cgroup/blkio.throttle.io_service_bytes':
                m = MagicMock()
                m.__enter__ = MagicMock(return_value=iter(io_content.split('\n')))
                m.__exit__ = MagicMock(return_value=False)
                return m
            raise FileNotFoundError()

        with patch('builtins.open', side_effect=mock_open_func):
            with patch('os.statvfs') as mock_statvfs:
                mock_statvfs.return_value = MagicMock(f_files=1000000, f_ffree=500000)
                result = get_container_disk_info()

        assert result.get('io_bytes') == 123456789

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_handles_disk_usage_exception(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.side_effect = Exception('Disk error')

        with patch('builtins.open', side_effect=FileNotFoundError()):
            with patch('os.statvfs') as mock_statvfs:
                mock_statvfs.return_value = MagicMock(f_files=1000000, f_ffree=500000)
                result = get_container_disk_info()

        assert 'total_bytes' not in result

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_handles_statvfs_exception(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)

        with patch('builtins.open', side_effect=FileNotFoundError()):
            with patch('os.statvfs', side_effect=Exception('statvfs error')):
                result = get_container_disk_info()

        assert 'total_inodes' not in result

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_handles_overall_exception(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.side_effect = Exception('Fatal error')

        with patch('builtins.open', side_effect=Exception('Fatal error')):
            with patch('os.statvfs', side_effect=Exception('Fatal error')):
                result = get_container_disk_info()

        assert result == {}

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_handles_overall_exception_with_job_id(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.side_effect = Exception('Fatal error')

        with patch('builtins.open', side_effect=Exception('Fatal error')):
            with patch('os.statvfs', side_effect=Exception('Fatal error')):
                result = get_container_disk_info(job_id='test-job')

        assert result == {}
        mock_logging.error.assert_called()

    @patch('handler.logging')
    @patch('handler.shutil.disk_usage')
    def test_no_disk_log_parts_without_job_id(self, mock_disk_usage, mock_logging):
        from handler import get_container_disk_info

        mock_disk_usage.side_effect = Exception('Disk error')

        with patch('builtins.open', side_effect=FileNotFoundError()):
            with patch('os.statvfs', side_effect=Exception('statvfs error')):
                result = get_container_disk_info()  # No job_id

        # Should hit the else branch for logging without job_id
        mock_logging.info.assert_called()


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

        mock_memory.return_value = {'available': 0.1}
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
        mock_disk.return_value = {'free_bytes': 100 * 1024}

        event = {'id': 'test-123', 'input': {'workflow': 'custom', 'payload': {}}}
        result = handler(event)

        assert 'error' in result
        assert 'disk' in result['error'].lower()


class TestHandlerSuccessPath:
    """Tests for handler success path."""

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    @patch('handler.os.path.exists')
    @patch('handler.os.remove')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake image data')
    def test_handler_success_with_output_image(
        self, mock_file, mock_remove, mock_exists, mock_get, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'test-prompt-123': {
                    'status': {'status_str': 'success', 'completed': True, 'messages': []},
                    'outputs': {
                        '9': {'images': [{'filename': 'test.png', 'type': 'output'}]}
                    }
                }
            }
        )

        mock_exists.return_value = True

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {
                    '9': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}
                }
            }
        }

        result = handler.handler(event)

        assert 'images' in result
        assert len(result['images']) == 1

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    @patch('handler.os.path.exists')
    @patch('handler.os.remove')
    def test_handler_success_with_temp_image_volume(
        self, mock_remove, mock_exists, mock_get, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'test-prompt-123': {
                    'status': {'status_str': 'success', 'completed': True, 'messages': []},
                    'outputs': {
                        '9': {'images': [{'filename': 'test.png', 'type': 'temp'}]}
                    }
                }
            }
        )

        mock_exists.return_value = True

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {
                    '9': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}
                }
            }
        }

        result = handler.handler(event)

        assert 'images' in result
        mock_remove.assert_called()

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    @patch('handler.os.path.exists')
    @patch('handler.os.remove')
    def test_handler_success_with_temp_image_tmp_dir(
        self, mock_remove, mock_exists, mock_get, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'test-prompt-123': {
                    'status': {'status_str': 'success', 'completed': True, 'messages': []},
                    'outputs': {
                        '9': {'images': [{'filename': 'test.png', 'type': 'temp'}]}
                    }
                }
            }
        )

        # First call (volume path) returns False, second call (/tmp) returns True
        mock_exists.side_effect = [False, True]

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {
                    '9': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}
                }
            }
        }

        result = handler.handler(event)

        assert 'images' in result
        mock_remove.assert_called()

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    @patch('handler.os.path.exists')
    @patch('handler.os.remove')
    def test_handler_temp_delete_error(
        self, mock_remove, mock_exists, mock_get, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'test-prompt-123': {
                    'status': {'status_str': 'success', 'completed': True, 'messages': []},
                    'outputs': {
                        '9': {'images': [{'filename': 'test.png', 'type': 'temp'}]}
                    }
                }
            }
        )

        mock_exists.return_value = True
        mock_remove.side_effect = Exception('Permission denied')

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {
                    '9': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}
                }
            }
        }

        result = handler.handler(event)

        assert 'images' in result
        mock_logging.error.assert_called()

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    @patch('handler.os.path.exists')
    @patch('handler.os.remove')
    def test_handler_temp_delete_error_tmp_path(
        self, mock_remove, mock_exists, mock_get, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'test-prompt-123': {
                    'status': {'status_str': 'success', 'completed': True, 'messages': []},
                    'outputs': {
                        '9': {'images': [{'filename': 'test.png', 'type': 'temp'}]}
                    }
                }
            }
        )

        mock_exists.side_effect = [False, True]
        mock_remove.side_effect = Exception('Permission denied')

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {
                    '9': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}
                }
            }
        }

        result = handler.handler(event)

        assert 'images' in result

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    @patch('handler.os.path.exists')
    def test_handler_low_memory_refresh(
        self, mock_exists, mock_get, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        # First call returns normal, second call (after processing) returns low memory
        mock_memory.side_effect = [
            {'available': 10.0},
            {'available': 0.5}
        ]
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'test-prompt-123': {
                    'status': {'status_str': 'success', 'completed': True, 'messages': []},
                    'outputs': {
                        '9': {'images': [{'filename': 'test.png', 'type': 'output'}]}
                    }
                }
            }
        )

        mock_exists.return_value = False

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {
                    '9': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}
                }
            }
        }

        result = handler.handler(event)

        assert result.get('refresh_worker') is True

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    def test_handler_no_outputs(
        self, mock_get, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'test-prompt-123': {
                    'status': {'status_str': 'success', 'completed': True, 'messages': []},
                    'outputs': {}
                }
            }
        )

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {}
            }
        }

        result = handler.handler(event)

        assert 'error' in result
        assert 'No output found' in result['error']

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    def test_handler_execution_error(
        self, mock_get, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'test-prompt-123': {
                    'status': {
                        'status_str': 'error',
                        'completed': False,
                        'messages': [
                            ['execution_error', {
                                'node_type': 'TestNode',
                                'exception_message': 'Test error'
                            }]
                        ]
                    },
                    'outputs': {}
                }
            }
        )

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {}
            }
        }

        result = handler.handler(event)

        assert 'error' in result
        assert 'TestNode' in result['error']

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    def test_handler_execution_error_without_details(
        self, mock_get, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'test-prompt-123': {
                    'status': {
                        'status_str': 'error',
                        'completed': False,
                        'messages': [
                            ['execution_error', {'other_field': 'value'}]
                        ]
                    },
                    'outputs': {}
                }
            }
        )

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {}
            }
        }

        result = handler.handler(event)

        assert 'error' in result

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    def test_handler_queue_error_with_json(
        self, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=500,
            json=lambda: {'error': 'Server error'},
            content=b'Server error'
        )

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {}
            }
        }

        result = handler.handler(event)

        assert 'error' in result
        assert '500' in result['error']

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    def test_handler_queue_error_without_json(
        self, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_response = MagicMock(status_code=500, content=b'Server error')
        mock_response.json.side_effect = Exception('Not JSON')
        mock_post.return_value = mock_response

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {}
            }
        }

        result = handler.handler(event)

        assert 'error' in result
        assert '500' in result['error']

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    @patch('handler.time.sleep')
    def test_handler_retries_history(
        self, mock_sleep, mock_get, mock_post,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        # First call returns empty, second returns result
        call_count = [0]
        def mock_json():
            call_count[0] += 1
            if call_count[0] == 1:
                return {}
            return {
                'test-prompt-123': {
                    'status': {'status_str': 'success', 'completed': True, 'messages': []},
                    'outputs': {}
                }
            }

        mock_get.return_value = MagicMock(status_code=200, json=mock_json)

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'custom',
                'payload': {}
            }
        }

        result = handler.handler(event)

        mock_sleep.assert_called()

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.get_workflow_payload')
    @patch('handler.send_post_request')
    @patch('handler.send_get_request')
    def test_handler_default_workflow(
        self, mock_get, mock_post, mock_workflow,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}
        mock_workflow.return_value = {}

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'prompt_id': 'test-prompt-123'}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'test-prompt-123': {
                    'status': {'status_str': 'success', 'completed': True, 'messages': []},
                    'outputs': {}
                }
            }
        )

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'default',
                'payload': {}
            }
        }

        result = handler.handler(event)

        mock_workflow.assert_called_with('txt2img', {})

    @patch('handler.logging')
    @patch('handler.get_container_memory_info')
    @patch('handler.get_container_cpu_info')
    @patch('handler.get_container_disk_info')
    @patch('handler.get_workflow_payload')
    def test_handler_workflow_load_error(
        self, mock_workflow,
        mock_disk, mock_cpu, mock_memory, mock_logging
    ):
        import handler

        mock_memory.return_value = {'available': 10.0}
        mock_cpu.return_value = {}
        mock_disk.return_value = {'free_bytes': 10 * 1024 * 1024 * 1024}
        mock_workflow.side_effect = Exception('File not found')

        event = {
            'id': 'test-123',
            'input': {
                'workflow': 'txt2img',
                'payload': {}
            }
        }

        result = handler.handler(event)

        assert 'error' in result
        assert 'refresh_worker' in result


class TestSnapLogHandler:
    """Tests for custom log handler."""

    def test_log_handler_formats_message_correctly(self, mock_runpod_logger):
        from handler import SnapLogHandler

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

            handler.emit(record)

    def test_log_handler_handles_format_args(self, mock_runpod_logger):
        from handler import SnapLogHandler

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

        handler.emit(record)

    def test_log_handler_handles_dict_args(self, mock_runpod_logger):
        from handler import SnapLogHandler

        handler = SnapLogHandler('test-app')
        handler.setFormatter(logging.Formatter('%(message)s'))

        # Use MagicMock because LogRecord doesn't accept dict as args directly
        record = MagicMock()
        record.msg = 'Test %(key)s message'
        record.args = {'key': 'value'}
        record.levelno = logging.INFO
        record.levelname = 'INFO'

        handler.emit(record)

    def test_log_handler_handles_dict_args_no_format(self, mock_runpod_logger):
        from handler import SnapLogHandler

        handler = SnapLogHandler('test-app')
        handler.setFormatter(logging.Formatter('%(message)s'))

        # Use MagicMock because LogRecord doesn't accept dict as args directly
        record = MagicMock()
        record.msg = 'Test message no format'
        record.args = {'key': 'value'}
        record.levelno = logging.INFO
        record.levelname = 'INFO'

        handler.emit(record)

    def test_log_handler_handles_format_error(self, mock_runpod_logger):
        from handler import SnapLogHandler

        handler = SnapLogHandler('test-app')
        handler.setFormatter(logging.Formatter('%(message)s'))

        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test %s %s message',
            args=('only_one',),
            exc_info=None
        )

        handler.emit(record)

    def test_log_handler_handles_no_msg_attr(self, mock_runpod_logger):
        from handler import SnapLogHandler

        handler = SnapLogHandler('test-app')
        handler.setFormatter(logging.Formatter('%(message)s'))

        record = MagicMock()
        del record.msg
        del record.args
        record.levelno = logging.INFO
        record.levelname = 'INFO'

        handler.emit(record)

    def test_log_handler_long_message_skips_runpod(self, mock_runpod_logger):
        from handler import SnapLogHandler

        handler = SnapLogHandler('test-app')
        handler.setFormatter(logging.Formatter('%(message)s'))

        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='x' * 1001,
            args=(),
            exc_info=None
        )

        handler.emit(record)

    def test_log_handler_without_job_id(self, mock_runpod_logger):
        from handler import SnapLogHandler

        with patch.dict(os.environ, {}, clear=True):
            if 'RUNPOD_JOB_ID' in os.environ:
                del os.environ['RUNPOD_JOB_ID']

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

            handler.emit(record)

    @patch('handler.requests.post')
    def test_log_handler_posts_to_api(self, mock_post, mock_runpod_logger):
        from handler import SnapLogHandler

        mock_post.return_value = MagicMock(status_code=200)

        with patch.dict(os.environ, {'LOG_API_ENDPOINT': 'http://test.com/log', 'LOG_API_TOKEN': 'token'}):
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

            handler.emit(record)

            mock_post.assert_called_once()

    @patch('handler.requests.post')
    def test_log_handler_handles_api_error(self, mock_post, mock_runpod_logger):
        from handler import SnapLogHandler

        mock_post.return_value = MagicMock(status_code=500)

        with patch.dict(os.environ, {'LOG_API_ENDPOINT': 'http://test.com/log', 'LOG_API_TOKEN': 'token'}):
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

            handler.emit(record)

    @patch('handler.requests.post')
    def test_log_handler_handles_api_timeout(self, mock_post, mock_runpod_logger):
        from handler import SnapLogHandler

        mock_post.side_effect = requests.Timeout()

        with patch.dict(os.environ, {'LOG_API_ENDPOINT': 'http://test.com/log', 'LOG_API_TOKEN': 'token'}):
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

            handler.emit(record)

    @patch('handler.requests.post')
    def test_log_handler_handles_api_exception(self, mock_post, mock_runpod_logger):
        from handler import SnapLogHandler

        mock_post.side_effect = Exception('Network error')

        with patch.dict(os.environ, {'LOG_API_ENDPOINT': 'http://test.com/log', 'LOG_API_TOKEN': 'token'}):
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

            handler.emit(record)

    def test_log_handler_emit_exception(self, mock_runpod_logger):
        from handler import SnapLogHandler

        handler = SnapLogHandler('test-app')
        handler.setFormatter(MagicMock(formatTime=MagicMock(side_effect=Exception('Format error'))))

        with patch.dict(os.environ, {'LOG_API_ENDPOINT': 'http://test.com/log'}):
            record = logging.LogRecord(
                name='test',
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg='Test message',
                args=(),
                exc_info=None
            )

            handler.emit(record)

    def test_log_handler_outer_exception(self, mock_runpod_logger):
        from handler import SnapLogHandler

        # Create handler but make rp_logger.info raise an exception
        # to trigger the outer exception handler
        handler = SnapLogHandler('test-app')
        handler.setFormatter(logging.Formatter('%(message)s'))
        handler.rp_logger.info = MagicMock(side_effect=RuntimeError('Outer exception'))

        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )

        # This should trigger the outer exception handler (lines 129-131)
        handler.emit(record)

        # The outer exception handler should have called rp_logger.error
        handler.rp_logger.error.assert_called()

    def test_log_handler_different_levels(self, mock_runpod_logger):
        from handler import SnapLogHandler

        handler = SnapLogHandler('test-app')
        handler.setFormatter(logging.Formatter('%(message)s'))

        for level in [logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL]:
            record = logging.LogRecord(
                name='test',
                level=level,
                pathname='',
                lineno=0,
                msg='Test message',
                args=(),
                exc_info=None
            )
            handler.emit(record)


class TestSetupLogging:
    """Tests for setup_logging function."""

    @patch('handler.SnapLogHandler')
    def test_setup_logging_configures_root_logger(self, mock_handler_class):
        from handler import setup_logging

        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler

        setup_logging()

        mock_handler.setFormatter.assert_called_once()
