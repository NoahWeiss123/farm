# farm-edge-agent

The local half of FARM. Pip-installable Python package that runs next to the arm, owns the control loop, the safety checks, the deterministic fallback, and the run-record writer. Talks to the cloud Worker over WebSocket and the arm over the xArm SDK.
