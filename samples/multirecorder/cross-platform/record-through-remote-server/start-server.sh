#!/bin/sh
# Starts a MultiRecorder camera server for the cameras connected to this
# computer. Pass the address to listen at; the default localhost:8090 accepts
# only local clients, so use this computer's IP address (for example
# 192.168.1.10:8090) when record-through-remote-server runs elsewhere.
# MultiRecorder comes from the XIMEA Software Package; set MULTIRECORDER to the
# MultiRecorder executable to use another copy. Press Ctrl+C to stop.
cd "$(dirname "$0")" || exit 1
MULTIRECORDER="${MULTIRECORDER:-/opt/XIMEA/examples/MultiRecorder/MultiRecorder}"
if [ ! -x "$MULTIRECORDER" ]; then
    echo "Error: MultiRecorder not found at $MULTIRECORDER. Install the XIMEA Software Package with its examples, or set MULTIRECORDER to the MultiRecorder executable." >&2
    exit 1
fi
exec "$MULTIRECORDER" --start-server "${1:-localhost:8090}"
