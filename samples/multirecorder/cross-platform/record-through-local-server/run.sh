#!/bin/sh
# Starts a camera server on this computer, records 10 images through it, then stops the server.
# MultiRecorder comes from the XIMEA Software Package; set MULTIRECORDER to the
# MultiRecorder executable to use another copy.
cd "$(dirname "$0")" || exit 1
MULTIRECORDER="${MULTIRECORDER:-/opt/XIMEA/examples/MultiRecorder/MultiRecorder}"
if [ ! -x "$MULTIRECORDER" ]; then
    echo "Error: MultiRecorder not found at $MULTIRECORDER. Install the XIMEA Software Package with its examples, or set MULTIRECORDER to the MultiRecorder executable." >&2
    exit 1
fi
# The camera server runs in the background and is stopped when the script exits.
"$MULTIRECORDER" --start-server localhost:8090 --log server-log.txt &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null' EXIT
# Give the server a moment to start listening.
sleep 1
"$MULTIRECORDER" --run setup.xml
