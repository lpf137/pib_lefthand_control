<!-- Copilot / AI agent instructions for quick onboarding and code edits -->

# Purpose
Short, actionable guidance for AI coding agents working in this repo (FTServo_Python).

# Big picture
- Core library: `scservo_sdk/` — low-level serial/packet handling for Feetech SCServo devices.
- Example apps: `sms_sts/`, `scscl/`, `hls/` — each contains small CLI examples (e.g., `ping.py`, `read.py`, `write.py`) that illustrate intended usage patterns.
- Data flow: examples instantiate `PortHandler` (serial), create a protocol handler (`sms_sts` / `scscl` / `hls`) and call packet-level APIs (`ping`, `read*`, `write*`).

# Key files and responsibilities
- `scservo_sdk/port_handler.py`: wraps pyserial, manages open/close, timeouts and tx timing.
- `scservo_sdk/protocol_packet_handler.py`: constructs/parses instruction/status packets, implements tx/rx lifecycle.
- `scservo_sdk/scservo_def.py`: constants (IDs, instructions, error bits) used across the SDK.
- `scservo_sdk/group_sync_read.py` / `group_sync_write.py`: batch read/write helpers used by examples.
- `sms_sts/ping.py` (example): canonical example for opening port, setting baudrate and pinging servo — use as pattern when adding new examples.

# Project-specific patterns and conventions
- Examples live alongside the core library; prefer adding new usage examples under `sms_sts/`, `scscl/` or `hls/` rather than modifying `scservo_sdk/` unless changing library behaviour.
- Error reporting uses numeric result codes and helper formatters: call `packetHandler.getTxRxResult(result)` and `packetHandler.getRxPacketError(error)` rather than free-form strings.
- Serial port strings are platform-specific in examples: Windows uses `"COMx"`, Linux uses `/dev/ttyUSB*`.
- The code targets Python 3 (README notes 3.5.3); avoid modern syntax that would break Python 3.5 if you intentionally maintain older compatibility.

# How to run & debug (discoverable from repo)
- Run examples from repository root (examples append parent path in code):

```bash
python3 sms_sts/ping.py
```

- If serial port access fails, check that `pyserial` is available (module `serial` is imported in `port_handler.py`).
- Use the example `ping.py` as a minimal smoke test: it demonstrates `PortHandler.openPort()`, `setBaudRate()` and `packetHandler.ping()`.

# Integration points & dependencies
- External runtime dependency: `pyserial` (imported as `serial` in `scservo_sdk/port_handler.py`).
- The library exposes a stable, thin API: create `PortHandler`, pass to protocol handlers (e.g., `sms_sts(portHandler)`), then call `ping/read/write` APIs.

# Editing guidance for AI agents
- When changing protocol logic, update `protocol_packet_handler.py` and keep tests/examples updated (modify `sms_sts/ping.py` to demonstrate API changes).
- Preserve existing function signatures (many downstream examples import `*` from `scservo_sdk`); avoid breaking top-level exports in `scservo_sdk/__init__.py`.
- Prefer minimal diffs: add helpers in `scservo_sdk/` only when they are reusable across examples.

# Useful examples to reference
- `sms_sts/ping.py` — start here for runtime steps and error handling.
- `scservo_sdk/port_handler.py` — shows serial configuration and timeout semantics.
- `scservo_sdk/protocol_packet_handler.py` — shows packet structure, checksum and tx/rx flow.

# What not to assume
- There are no automated tests / CI config discoverable in the repo. Validate changes by running the existing examples on a machine with serial access to servos.

---
If anything here is unclear or you want the instructions tailored (e.g., stricter compatibility rules or a short contributor checklist), tell me what to emphasize and I will iterate.
