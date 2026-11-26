#!/bin/sh
# Starts a camera server on this computer, records 10 images through it, then stops the server.
# MultiRecorder comes from the XIMEA Software Package. The script runs
# $MULTIRECORDER if set, else the copy under $XIMEA_SP_PATH if that is set,
# else /opt/XIMEA/examples/MultiRecorder/MultiRecorder.
cd "$(dirname "$0")" || exit 1
if [ -z "${MULTIRECORDER:-}" ]; then
    if [ -n "${XIMEA_SP_PATH:-}" ]; then
        # The SP names the folder examples/ or Examples/ depending on the package.
        MULTIRECORDER="$XIMEA_SP_PATH/examples/MultiRecorder/MultiRecorder"
        if [ ! -x "$MULTIRECORDER" ] && [ -x "$XIMEA_SP_PATH/Examples/MultiRecorder/MultiRecorder" ]; then
            MULTIRECORDER="$XIMEA_SP_PATH/Examples/MultiRecorder/MultiRecorder"
        fi
    else
        MULTIRECORDER=/opt/XIMEA/examples/MultiRecorder/MultiRecorder
    fi
fi
if [ ! -x "$MULTIRECORDER" ]; then
    echo "Error: MultiRecorder not found at $MULTIRECORDER. Install the XIMEA Software Package with its examples, set XIMEA_SP_PATH to its install folder, or set MULTIRECORDER to the MultiRecorder executable." >&2
    exit 1
fi
# The camera server runs in the background and is stopped when the script exits.
"$MULTIRECORDER" --start-server localhost:8090 --log server-log.txt &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null' EXIT
# Give the server a moment to start listening.
sleep 1
"$MULTIRECORDER" --run setup.xml
