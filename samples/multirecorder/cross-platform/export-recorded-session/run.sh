#!/bin/sh
# Records 10 images into a saved session, then exports the session as TIFF files.
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
if ! "$MULTIRECORDER" --run setup.xml; then
    echo "Error: recording failed; skipping the export." >&2
    exit 1
fi
exec "$MULTIRECORDER" --export-tiffs recorded-session.xirecsession --export-tiffs-folder exported
