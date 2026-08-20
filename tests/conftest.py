"""Global pytest configuration for deterministic, offline test teardown."""

import os

# Gradio creates non-daemon version-check/telemetry threads by default. On
# networks where api.gradio.app is unreachable, those threads can remain in a
# socket connect and keep Python alive after pytest has printed its summary.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
