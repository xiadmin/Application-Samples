#!/bin/sh
# Records 10 images into a saved session, then exports the session as TIFF files.
# MultiRecorder comes from the XIMEA Software Package; set MULTIRECORDER to the
# MultiRecorder executable to use another copy.
cd "$(dirname "$0")" || exit 1
MULTIRECORDER="${MULTIRECORDER:-/opt/XIMEA/examples/MultiRecorder/MultiRecorder}"
if [ ! -x "$MULTIRECORDER" ]; then
    echo "Error: MultiRecorder not found at $MULTIRECORDER. Install the XIMEA Software Package with its examples, or set MULTIRECORDER to the MultiRecorder executable." >&2
    exit 1
fi
if ! "$MULTIRECORDER" --run setup.xml; then
    echo "Error: recording failed; skipping the export." >&2
    exit 1
fi
exec "$MULTIRECORDER" --export-tiffs recorded-session.xirecsession --export-tiffs-folder exported
