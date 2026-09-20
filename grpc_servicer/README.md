# smg-grpc-servicer

gRPC servicer implementations for LLM inference engines. Supports vLLM, MLX, TokenSpeed, and SGLang.

## Installation

For vLLM:

```bash
pip install smg-grpc-servicer[vllm]
```

For MLX:

```bash
pip install smg-grpc-servicer[mlx]
```

For TokenSpeed, install the TokenSpeed runtime first, then install the servicer bridge:

```bash
pip install smg-grpc-servicer
```

For SGLang:

```bash
pip install smg-grpc-servicer[sglang]
```

## Usage

### vLLM

```bash
vllm serve meta-llama/Llama-2-7b-hf --grpc
```

### MLX

```bash
python -m smg_grpc_servicer.mlx --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --port 50051
```

### TokenSpeed

```bash
python -m smg_grpc_servicer.tokenspeed --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --port 50051
```

### SGLang

```bash
sglang serve --model-path meta-llama/Llama-2-7b-hf --grpc-mode
```

## Architecture

```
smg-grpc-servicer[vllm]    ──optional dep──>  vllm       (lazy import)
smg-grpc-servicer[mlx]     ──optional dep──>  mlx-lm     (lazy import)
smg-grpc-servicer          ──external runtime──>  tokenspeed (lazy import)
smg-grpc-servicer[sglang]  ──optional dep──>  sglang     (lazy import)
smg-grpc-servicer          ──depends on────>  smg-grpc-proto  (hard dependency)
vllm                       ──optional──────>  smg-grpc-servicer (via vllm serve --grpc)
sglang                     ──optional──────>  smg-grpc-servicer (via --grpc-mode)
```

Backend dependencies are isolated via extras or runtime installs to avoid conflicts between vLLM, MLX, TokenSpeed, and SGLang.

## Two-tier Worker control plane

The SGLang and TokenSpeed servicers can embed the Rust SMG Worker
(`smg.worker.v1` WorkerControl) so a fleet-level SMG Router discovers, health
probes and drains them through it. The control plane is configured entirely
through `SMG_WORKER_*` environment variables, read by
`smg_grpc_servicer.worker_control_lifecycle`. It requires the `smg` Python
package (the extension is imported only when enabled), and every variable is
ignored unless `SMG_WORKER_CONTROL_BIND_ADDRESS` is set.

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `SMG_WORKER_CONTROL_BIND_ADDRESS` | enable switch | unset (control plane off) | `host:port` the WorkerControl gRPC listener binds. Setting it enables the control plane; every other variable is read only then. |
| `SMG_WORKER_ENGINE_ENDPOINT` | yes, once enabled | none | Address the Worker advertises and dials for the engine: `grpc://host:port` of this servicer, or the `ipc://` socket URL with the `zmq` transport. |
| `SMG_WORKER_ID` | no | the hostname | Identity the Worker reports to the Router. |
| `SMG_WORKER_INSTANCE_ID` | no | `<worker id>-<random hex>` | Per-process instance id; set it to keep a stable id across restarts. |
| `SMG_WORKER_ZONE` | no | empty | Zone/topology label reported to the Router. |
| `SMG_WORKER_INFERENCE_ENABLED` | no | `false` | Boolean (`1/true/yes/on` or `0/false/no/off`). Serves WorkerInference (tokenized Generate/Abort) from the Worker via the engine transport. Valid only for `vllm` and `tokenspeed`; the SGLang servicer rejects it at startup. |
| `SMG_WORKER_ENGINE_TRANSPORT` | no | `grpc` | `grpc` (the engine's own gRPC server) or `zmq` (msgpack ZMQ IPC). |
| `SMG_WORKER_ZMQ_HANDSHAKE_ADDRESS` | no | derived from the `ipc://` endpoint | `tcp://host:port` the Worker binds for the ZMQ engine handshake when the engine dials a fixed, pre-agreed address. Used only with the `zmq` transport. |
| `SMG_WORKER_ENGINE_COUNT` | no | `1` | Positive integer: engines sharing one ZMQ socket set (the engine-level data-parallel size). Used only with the `zmq` transport. |

Any invalid value (a non-boolean `SMG_WORKER_INFERENCE_ENABLED`, a
non-positive `SMG_WORKER_ENGINE_COUNT`, a missing endpoint, or a combination
the extension rejects) fails the servicer at startup instead of coming up
without a control plane.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for local development setup, CI, and release workflows.
