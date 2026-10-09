#!/bin/sh
# Records 10 images from each of three cameras and exports them as TIFF files.
# MultiRecorder comes from the XIMEA Software Package; set MULTIRECORDER to the
# MultiRecorder executable to use another copy.
cd "$(dirname "$0")" || exit 1
MULTIRECORDER="${MULTIRECORDER:-/opt/XIMEA/examples/MultiRecorder/MultiRecorder}"
if [ ! -x "$MULTIRECORDER" ]; then
    echo "Error: MultiRecorder not found at $MULTIRECORDER. Install the XIMEA Software Package with its examples, or set MULTIRECORDER to the MultiRecorder executable." >&2
    exit 1
fi
exec "$MULTIRECORDER" --run setup.xml
