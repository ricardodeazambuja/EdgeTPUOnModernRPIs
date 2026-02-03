"""CLI client for interacting with the Edge TPU inference service."""

import argparse
import json
import subprocess
import sys

import numpy as np

from .client import EdgeTPUClient, EdgeTPUBusyError

SERVICE_NAME = "edgetpu"


def _connect(args):
    """Create a client connection, handling errors."""
    socket_path = getattr(args, "socket", None) or "/tmp/edgetpu.sock"
    try:
        return EdgeTPUClient(socket_path=socket_path)
    except (ConnectionRefusedError, FileNotFoundError) as e:
        _error(f"Cannot connect to service: {e}", args)
        sys.exit(1)


def _output_json(data, args):
    """Print JSON to stdout."""
    print(json.dumps(data, indent=2))


def _output_array(arr, args, output_path=None):
    """Output a numpy array: save to file or print summary."""
    if output_path:
        np.save(output_path, arr)
        if getattr(args, "json_output", False):
            _output_json({"saved": output_path, "shape": list(arr.shape), "dtype": str(arr.dtype)}, args)
        else:
            print(f"Saved to {output_path} (shape={arr.shape}, dtype={arr.dtype})")
    else:
        if getattr(args, "json_output", False):
            _output_json({
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "values": arr.flatten()[:20].tolist(),
                "truncated": arr.size > 20,
            }, args)
        else:
            print(f"shape: {arr.shape}")
            print(f"dtype: {arr.dtype}")
            flat = arr.flatten()
            if flat.size <= 20:
                print(f"values: {flat.tolist()}")
            else:
                print(f"values (first 20): {flat[:20].tolist()}")
                print(f"  ... ({arr.size} total elements)")


def _error(msg, args):
    """Print error to stderr."""
    if getattr(args, "json_output", False):
        print(json.dumps({"error": msg}), file=sys.stderr)
    else:
        print(f"error: {msg}", file=sys.stderr)


def cmd_ping(args):
    """Check if the service is alive."""
    client = _connect(args)
    try:
        result = client.ping()
        _output_json(result, args)
    except Exception as e:
        _error(str(e), args)
        sys.exit(1)
    finally:
        client.close()


def cmd_status(args):
    """Show systemd service status."""
    try:
        result = subprocess.run(
            ["systemctl", "status", SERVICE_NAME, "--no-pager"],
            capture_output=True, text=True,
        )
        if getattr(args, "json_output", False):
            _output_json({
                "active": result.returncode == 0,
                "output": result.stdout.strip(),
            }, args)
        else:
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode if result.returncode in (0, 3) else 1)
    except FileNotFoundError:
        _error("systemctl not found", args)
        sys.exit(1)


def cmd_load_model(args):
    """Load a model onto the TPU."""
    client = _connect(args)
    try:
        result = client.load_model(args.model_path)
        _output_json(result, args)
    except EdgeTPUBusyError as e:
        _error(f"busy: {e}", args)
        sys.exit(1)
    except Exception as e:
        _error(str(e), args)
        sys.exit(1)
    finally:
        client.close()


def cmd_infer(args):
    """Run inference with input from a .npy file."""
    try:
        input_data = np.load(args.input_npy)
    except Exception as e:
        _error(f"Cannot load input: {e}", args)
        sys.exit(1)

    client = _connect(args)
    try:
        client.load_model(args.model_path)
        result = client.infer(input_data)
        _output_array(result, args, output_path=args.output)
    except EdgeTPUBusyError as e:
        _error(f"busy: {e}", args)
        sys.exit(1)
    except Exception as e:
        _error(str(e), args)
        sys.exit(1)
    finally:
        client.close()


def cmd_embedding(args):
    """Extract embedding from a .npy input."""
    try:
        input_data = np.load(args.input_npy)
    except Exception as e:
        _error(f"Cannot load input: {e}", args)
        sys.exit(1)

    shape = None
    if args.shape:
        try:
            shape = [int(x) for x in args.shape.split(",")]
        except ValueError:
            _error(f"Invalid shape: {args.shape} (expected comma-separated ints)", args)
            sys.exit(1)

    client = _connect(args)
    try:
        client.load_model(args.model_path)
        result = client.get_embedding(input_data, embedding_shape=shape)
        _output_array(result, args, output_path=args.output)
    except EdgeTPUBusyError as e:
        _error(f"busy: {e}", args)
        sys.exit(1)
    except Exception as e:
        _error(str(e), args)
        sys.exit(1)
    finally:
        client.close()


def cmd_pipeline(args):
    """Run a multi-model pipeline on a .npy input."""
    try:
        input_data = np.load(args.input)
    except Exception as e:
        _error(f"Cannot load input: {e}", args)
        sys.exit(1)

    client = _connect(args)
    try:
        result = client.pipeline(args.models, input_data)
        _output_array(result, args, output_path=args.output)
    except EdgeTPUBusyError as e:
        _error(f"busy: {e}", args)
        sys.exit(1)
    except Exception as e:
        _error(str(e), args)
        sys.exit(1)
    finally:
        client.close()


def cmd_rescan_tpus(args):
    """Trigger a rescan for Edge TPU devices."""
    client = _connect(args)
    try:
        result = client.rescan_tpus()
        _output_json(result, args)
    except Exception as e:
        _error(str(e), args)
        sys.exit(1)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        prog="edgetpu-cli",
        description="Command-line client for the Edge TPU inference service",
    )
    parser.add_argument(
        "--socket", default="/tmp/edgetpu.sock",
        help="Path to the service Unix socket (default: /tmp/edgetpu.sock)",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Machine-readable JSON output for all commands",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ping
    sub.add_parser("ping", help="Check if the service is alive")

    # status
    sub.add_parser("status", help="Show systemd service status")

    # load-model
    p_load = sub.add_parser("load-model", help="Load a model onto the TPU")
    p_load.add_argument("model_path", help="Path to *_edgetpu.tflite model file")

    # infer
    p_infer = sub.add_parser("infer", help="Run inference on a .npy input")
    p_infer.add_argument("model_path", help="Path to *_edgetpu.tflite model file")
    p_infer.add_argument("input_npy", help="Path to input .npy file")
    p_infer.add_argument("-o", "--output", help="Save output to .npy file")

    # embedding
    p_embed = sub.add_parser("embedding", help="Extract embedding from a .npy input")
    p_embed.add_argument("model_path", help="Path to *_edgetpu.tflite model file")
    p_embed.add_argument("input_npy", help="Path to input .npy file")
    p_embed.add_argument("-o", "--output", help="Save output to .npy file")
    p_embed.add_argument("--shape", help="Embedding layer shape (comma-separated, e.g. 1,1280)")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Run a multi-model pipeline on a .npy input")
    p_pipe.add_argument("models", nargs="+", help="Model paths in pipeline order (>= 2)")
    p_pipe.add_argument("--input", required=True, help="Path to input .npy file")
    p_pipe.add_argument("-o", "--output", help="Save output to .npy file")

    # rescan-tpus
    sub.add_parser("rescan-tpus", help="Re-scan for Edge TPU devices (hot-plug support)")

    args = parser.parse_args()

    commands = {
        "ping": cmd_ping,
        "status": cmd_status,
        "load-model": cmd_load_model,
        "infer": cmd_infer,
        "embedding": cmd_embedding,
        "pipeline": cmd_pipeline,
        "rescan-tpus": cmd_rescan_tpus,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
