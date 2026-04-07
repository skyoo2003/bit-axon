# Bit-Axon macOS App

A native SwiftUI application for running Bit-Axon inference on Apple Silicon. The app wraps the same 24-layer hybrid architecture as the Python package (Axon-SSM, sliding window attention, shared-expert MoE) in Swift and MLX-Swift, with a chat interface, real-time streaming, GPU monitoring, and drag-and-drop fine-tuning.

## Requirements

| Requirement | Version |
|---|---|
| macOS | 14 (Sonoma) or later |
| Xcode | 15 or later |
| Swift Package Manager | Bundled with Xcode |
| Hardware | Apple Silicon (M1 or later) |

## Dependencies

Declared in `Package.swift` and resolved automatically by SPM:

| Package | Version | Purpose |
|---|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | 0.29.1 to <0.30.0 | Core MLX bindings for GPU-accelerated tensors |
| [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) | 2.21.2+ | Shared LLM utilities (MLXLMCommon) |

!!! note
    Both dependencies resolve through Swift Package Manager. There is nothing to install manually. Xcode fetches them on first build, or `swift package resolve` works from the command line.

## Installation

### Command line

```bash
cd BitAxonApp
swift build
```

### Xcode

```bash
open BitAxonApp.xcodeproj
# or, if you prefer the SPM workspace:
open Package.swift
```

Then press **Cmd+R** to build and run. The scheme defaults to `BitAxonApp`.

### Running tests

```bash
cd BitAxonApp
swift test --filter EquivalenceTests
```

This runs the cross-language equivalence suite that validates Python-to-Swift numerical parity for every layer type. See [Equivalence Tests](#equivalence-tests) for details.

## Features

### Real-time token streaming

Tokens arrive one at a time as the model generates. The assistant message updates live, and the toolbar displays tokens-per-second throughput as well as time-to-first-token latency.

### GPU memory monitoring

The `DeviceStat` service polls `GPU.snapshot()` every two seconds and surfaces four metrics in the toolbar: active memory, cache memory, peak memory, and the configured memory limit. It also reads the SoC die temperature via `powermetrics` when running with elevated privileges.

### Drag-and-drop fine-tuning

The Fine-Tune view accepts a JSONL data file via drag-and-drop (or a file picker), then shells out to the `bit-axon train` CLI under the hood. The app streams the training log in real time, parsing step count and loss values to display a progress bar. Training runs as a subprocess, so the GUI stays responsive.

### Chat interface with conversation history

A `NavigationSplitView` layout puts a sidebar (model controls, fine-tune link, metrics toggle) next to the main chat area. Messages accumulate in a scrollable list with user/assistant roles. The input field clears on send, and a "Clear Chat" button wipes the history.

## Architecture

### Directory layout

```
BitAxonApp/
├── BitAxonApp.swift          # @main entry point, sets GPU cache limit
├── ContentView.swift         # NavigationSplitView: sidebar + chat detail
├── Models/
│   ├── BitAxonConfig.swift   # Codable config struct (mirrors Python BitAxonConfig)
│   ├── BitAxonModel.swift    # 24-layer model, dispatches to block variants
│   ├── BitAxonKVCache.swift  # KV cache for SWA layers
│   └── Layers/
│       ├── AxonSSM.swift     # Mamba-style state space model
│       ├── AxonSWA.swift     # Sliding window attention (4K window)
│       ├── AxonMoE.swift     # Shared-expert mixture of experts
│       ├── AxonRMSNorm.swift # RMS normalization
│       └── AxonBlocks.swift  # Three block types: SSM, SWA+MoE, SSM+MoE
├── ViewModels/
│   ├── ChatViewModel.swift   # Generation loop, message state, mock tokenizer
│   └── DeviceStat.swift      # GPU memory polling, temperature reading
├── Views/
│   ├── ChatView.swift        # Message list + input area
│   ├── MessageRow.swift      # Single message bubble
│   ├── PromptInputView.swift # Text field with send button
│   ├── MetricsView.swift     # Token speed, TTFT, GPU stats toolbar
│   └── FineTuneView.swift    # Training config + log streaming
└── Services/
    ├── ModelService.swift    # Model loading (from directory or default config)
    ├── FineTuneBridge.swift  # Subprocess bridge to bit-axon CLI
    └── BitAxonRegistry.swift # Factory for creating models with default config
```

### Swift model ports

Each Python model class has a direct Swift counterpart. The config uses `CodingKeys` to map between Swift camelCase and the Python `config.json` snake_case keys, so the same config file loads in both languages.

| Python class | Swift class | Notes |
|---|---|---|
| `BitAxonConfig` | `BitAxonConfig` | Same defaults, `Codable` for JSON |
| `BitAxonModel` | `BitAxonModel` | `Module` subclass, `LayerCache` enum for per-layer cache |
| `BitAxonKVCache` | `BitAxonKVCache` | Only used by SWA layers (9-16) |

### Layer ports

| Python module | Swift file | Key types |
|---|---|---|
| `axon_ssm.py` | `AxonSSM.swift` | `AxonSSM` (SSM layer with conv1d, state vectors) |
| `swa.py` | `AxonSWA.swift` | `AxonSWA` (sliding window attention) |
| `moe.py` | `AxonMoE.swift` | `AxonSharedExpertMoE` (gate, 8 experts, shared expert) |
| `rms_norm.py` | `AxonRMSNorm.swift` | `AxonRMSNorm` |
| `block.py` | `AxonBlocks.swift` | `AxonSSMBlock`, `AxonSWAMoEBlock`, `AxonSSMMoEBlock` |

The model's `getLayerType` function mirrors the Python sandwich layout: layers 0-7 are pure SSM, 8-15 are SWA+MoE, and 16-23 are SSM+MoE.

### ViewModel layer

**`ChatViewModel`** owns the generation loop. It holds the message array, drives the autoregressive decode on a detached `Task`, and streams partial text back to the main actor. A placeholder `MockTokenizer` handles encode/decode for now; a real tokenizer (Qwen2.5 compatible) will replace it.

**`DeviceStat`** starts a repeating timer that reads `GPU.snapshot()` and optionally `powermetrics`. All properties are `@MainActor` so SwiftUI binds directly.

### Service layer

**`ModelService`** manages model lifecycle. It loads from a directory (reads `config.json`, constructs the model) or falls back to a default config via `BitAxonModelRegistry`. State transitions through `idle`, `loading(progress:)`, `ready`, and `failed(String)`.

**`FineTuneBridge`** discovers the `bit-axon` CLI in `PATH` (or common fallback paths), launches it as a `Process` with training arguments, and streams stdout/stderr back to the UI. It parses step and loss from log lines to drive the progress bar.

**`BitAxonRegistry`** is a thin factory that creates `BitAxonModel` instances from a config.

## Equivalence Tests

The test target `EquivalenceTests` lives at `BitAxonApp/Tests/EquivalenceTests/` and validates that every Swift layer produces the same numerical output as its Python counterpart.

### How it works

1. `export_reference.py` (excluded from the build) runs each Python layer with deterministic weights and inputs, then serializes the inputs, weights, and outputs to JSON files in `EquivalenceTestSupport/reference/`.
2. Each Swift test loads the reference JSON, reconstructs the layer with the same weights, runs the forward pass, and asserts the output tensors match within tolerance.

### What is tested

| Test | Layer | Tolerance |
|---|---|---|
| `testRMSNorm` | `AxonRMSNorm` | 1e-3 |
| `testAxonSSM` | `AxonSSM` (including conv cache and SSM state) | 1e-2 |
| `testAxonSWA` | `AxonSWA` | 1e-3 |
| `testAxonMoE` | `AxonSharedExpertMoE` | 5e-2 |

The MoE tolerance is looser because the expert routing and gating introduce additional floating-point divergence between PyTorch and MLX backends.

### Running the tests

```bash
# All equivalence tests
cd BitAxonApp && swift test --filter EquivalenceTests

# A single test
cd BitAxonApp && swift test --filter EquivalenceTests/testAxonSSM
```

### Regenerating reference data

After changing a Python layer, regenerate the reference tensors:

```bash
cd BitAxonApp/Tests/EquivalenceTests/EquivalenceTestSupport
python export_reference.py
```

## Building and Running

### Quick start from the command line

```bash
# Clone and enter the app directory
cd bit-axon/BitAxonApp

# Resolve dependencies
swift package resolve

# Build
swift build

# Run (opens the SwiftUI window)
.build/debug/BitAxonApp
```

### From Xcode

1. Open `BitAxonApp.xcodeproj` (or `Package.swift` for the SPM workspace).
2. Select the **My Mac** destination.
3. Press **Cmd+R**.

### Loading a model

On launch, the app shows a sidebar with a "Load Model" button. Tap it to instantiate a model with the default config. To load a specific checkpoint, drop the model directory onto the app (this feature uses `ModelService.loadFromDirectory`, which expects a `config.json` at the root of the directory).

### Monitoring performance

Toggle "Show Metrics" in the sidebar. The toolbar shows:

- **Tokens/sec**: real-time generation throughput
- **TTFT**: time-to-first-token in milliseconds
- **GPU Used / Cache / Peak / Limit**: memory figures in MB
- **SoC Temp**: die temperature (requires `sudo` for `powermetrics` access)

### Fine-tuning

1. Click "Fine-Tune" in the sidebar.
2. Drag a JSONL training file onto the drop zone, or click to browse.
3. Adjust hyperparameters (learning rate, LoRA rank, batch size, etc.).
4. Click "Start Training."

The app shells out to `bit-axon train` and streams the log. The `bit-axon` CLI must be installed (`pip install bit-axon`) and discoverable in `PATH`.
