"""
Edge TPU client library.
Works with any Python version. Communicates with edgetpu_service via Unix socket.
"""

import socket
import struct
import json
import numpy as np

SOCKET_PATH = "/tmp/edgetpu.sock"


class EdgeTPUError(Exception):
    """Raised when the service reports an error (bad model, no TPU, etc.)."""
    pass


class EdgeTPUBusyError(EdgeTPUError):
    """Raised when the server's inference queue is full."""
    pass


class EdgeTPUClient:
    """
    Client for Edge TPU inference service.

    Usage:
        client = EdgeTPUClient()
        client.load_model("/path/to/model_edgetpu.tflite")

        # For classification/detection
        output = client.infer(image_array)

        # For feature extraction
        embedding = client.get_embedding(image_array)

        client.close()
    """

    def __init__(self, socket_path=SOCKET_PATH):
        self.socket_path = socket_path
        self.sock = None
        self._connect()

    def _connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)
        self.sock.settimeout(30.0)

    def _recv_exact(self, n):
        buf = bytearray(n)
        view = memoryview(buf)
        pos = 0
        try:
            while pos < n:
                nbytes = self.sock.recv_into(view[pos:])
                if nbytes == 0:
                    raise ConnectionError("Socket closed")
                pos += nbytes
        except socket.timeout:
            raise EdgeTPUError("Operation timed out")
        return bytes(buf)

    def _send_command(self, header, data=None):
        header_bytes = json.dumps(header).encode('utf-8')
        self.sock.sendall(struct.pack('!I', len(header_bytes)) + header_bytes)
        if data is not None:
            self.sock.sendall(data)

    def _recv_json(self):
        resp_len = struct.unpack('!I', self._recv_exact(4))[0]
        result = json.loads(self._recv_exact(resp_len).decode('utf-8'))
        if isinstance(result, dict) and 'error' in result:
            if result['error'] == 'server_busy':
                raise EdgeTPUBusyError(result.get('message', 'Server busy'))
            raise EdgeTPUError(result.get('message', result['error']))
        return result

    def _recv_array_response(self):
        """Receive response that might be numpy array or JSON error."""
        data_size = struct.unpack('!I', self._recv_exact(4))[0]

        if data_size == 0:  # JSON error response
            return self._recv_json()

        # Numpy array response
        header_len = struct.unpack('!I', self._recv_exact(4))[0]
        header = json.loads(self._recv_exact(header_len).decode('utf-8'))
        array_bytes = self._recv_exact(data_size)

        return np.frombuffer(array_bytes, dtype=header['dtype']).reshape(header['shape'])

    def ping(self):
        """Check if service is alive."""
        self._send_command({'command': 'ping'})
        return self._recv_json()

    def load_model(self, model_path):
        """
        Load model onto Edge TPU.

        Args:
            model_path: Path to *_edgetpu.tflite file

        Returns:
            dict with input_shape, input_dtype, output_shape
        """
        self._send_command({'command': 'load_model', 'model_path': model_path})
        return self._recv_json()

    def infer(self, input_array):
        """
        Run inference.

        Args:
            input_array: numpy array matching model's expected input shape/dtype

        Returns:
            numpy array of model output
        """
        input_array = np.ascontiguousarray(input_array)
        self._send_command({
            'command': 'infer',
            'shape': list(input_array.shape),
            'dtype': str(input_array.dtype),
            'data_size': input_array.nbytes
        }, input_array)
        return self._recv_array_response()

    def detect(self, input_array, score_threshold=0.5, top_k=25):
        """
        Run SSD detection model and return parsed results.

        Works with SSD models that include TFLite_Detection_PostProcess
        (the standard Coral SSD models).

        Args:
            input_array: numpy array matching model's expected input shape/dtype
            score_threshold: minimum confidence to include a detection
            top_k: maximum number of detections to return

        Returns:
            list of dicts with keys: class_id, score, bbox (ymin, xmin, ymax, xmax)
        """
        input_array = np.ascontiguousarray(input_array)
        self._send_command({
            'command': 'detect',
            'shape': list(input_array.shape),
            'dtype': str(input_array.dtype),
            'data_size': input_array.nbytes
        }, input_array)
        result = self._recv_json()

        num_outputs = result.get('num_outputs', 0)
        if num_outputs < 4:
            raise RuntimeError(
                f"Expected 4 output tensors from SSD model, got {num_outputs}. "
                "Make sure the model includes TFLite_Detection_PostProcess."
            )

        boxes = np.array(result['output_0']).reshape(result['output_0_shape'])
        classes = np.array(result['output_1']).reshape(result['output_1_shape'])
        scores = np.array(result['output_2']).reshape(result['output_2_shape'])
        count = int(np.array(result['output_3']).flatten()[0])

        detections = []
        for i in range(min(count, top_k)):
            score = float(scores[0][i])
            if score < score_threshold:
                continue
            detections.append({
                'class_id': int(classes[0][i]),
                'score': score,
                'bbox': {
                    'ymin': float(boxes[0][i][0]),
                    'xmin': float(boxes[0][i][1]),
                    'ymax': float(boxes[0][i][2]),
                    'xmax': float(boxes[0][i][3]),
                }
            })

        return detections

    def get_embedding(self, input_array, embedding_shape=None):
        """
        Get embedding from intermediate layer.

        Args:
            input_array: numpy array matching model's expected input
            embedding_shape: shape to look for, default [1, 1280]

        Returns:
            numpy array of embedding
        """
        if embedding_shape is None:
            embedding_shape = [1, 1280]

        input_array = np.ascontiguousarray(input_array)
        self._send_command({
            'command': 'embedding',
            'shape': list(input_array.shape),
            'dtype': str(input_array.dtype),
            'data_size': input_array.nbytes,
            'embedding_shape': embedding_shape
        }, input_array)
        return self._recv_array_response()

    def pipeline(self, models, input_array):
        """Run a multi-model pipeline server-side.

        Each model's output feeds into the next model's input.
        Intermediate tensors stay on the server.

        Args:
            models: list of paths to *_edgetpu.tflite files (>= 2)
            input_array: numpy array matching the first model's expected input

        Returns:
            numpy array of the final model's output
        """
        if not isinstance(models, list) or len(models) < 2:
            raise ValueError("pipeline requires a list of >= 2 model paths")
        input_array = np.ascontiguousarray(input_array)
        self._send_command({
            'command': 'pipeline',
            'models': models,
            'shape': list(input_array.shape),
            'dtype': str(input_array.dtype),
            'data_size': input_array.nbytes,
        }, input_array)
        return self._recv_array_response()

    def rescan_tpus(self):
        """Tell the service to re-scan for Edge TPU devices."""
        self._send_command({'command': 'rescan_tpus'})
        return self._recv_json()

    def close(self):
        """Close connection."""
        try:
            self._send_command({'command': 'quit'})
        except Exception:
            pass
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
