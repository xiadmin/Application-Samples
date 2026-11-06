# ffs-parameter-preset

Saves a name=value parameter preset into the camera Flash File System (FFS) file User1, shows it, and applies it back to the camera.

| Item | Value |
|------|-------|
| Category | Firmware / file access / LUT |
| OS platform | Cross-Platform |
| Hardware platform | Cross-platform |
| API type | xiAPI Python |
| Language | Python |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| XIMEA SDK | 4.32+ |
| Libraries | None |
| Python | 3.11+ |
| Library | None |

---

## Build

No build step is required for this Python sample.

---

## Run

The sample has three commands. Without a command it runs `show`. Add `-s SERIAL` before the command to choose a camera by serial number; otherwise it opens the first camera.

```bash
cd samples/xiapi-python/cross-platform/ffs-parameter-preset
python main.py save exposure=10000 gain=2.0
python main.py show
python main.py load
```

`exposure` and `gain` are available on every camera family. Which other parameters you can store depends on the camera model; see [Camera-family-specific presets](#camera-family-specific-presets).

- `save name=value [name=value ...]` checks every pair against the xiAPI parameter list, applies the pairs to the camera, writes them to the FFS file `User1` and reads the file back to verify it.
- `show` prints the stored preset.
- `load` reads the preset and applies the pairs to the camera in the stored order.

Use it in your own application by calling `load_preset(cam)` right after opening the camera, so camera-specific settings such as a per-camera digital gain go with the camera to any PC.

Preset rules:

- Names are xiAPI parameter strings, such as `gain`, `exposure` or `gain_selector`. Enumerated values are written by name, such as `XI_GAIN_SELECTOR_DIGITAL_ALL`. Booleans accept `0`, `1`, `true` or `false`.
- Pairs are applied in the order they were saved, so a selector must come before the parameter it selects (`gain_selector` before `gain`).
- Some parameters reset others; for example `imgdataformat` or `downsampling` can change gain or ROI. Put them first in the preset, or load the preset after your application has set them.
- `save` replaces the whole file. To change one value, save the complete preset again.
- `save` applies every pair to the camera before writing, so a preset that the camera rejects is never stored. The sample does not check the camera model; the camera itself decides which parameters it accepts.

The stored file is plain text:

```text
# XIMEA parameter preset
exposure=10000
gain=2.0
```

### Camera-family-specific presets

Some parameters exist only on some camera families. For example, a per-camera digital gain correction uses `gain_selector`:

```bash
python main.py save gain_selector=XI_GAIN_SELECTOR_DIGITAL_ALL gain=0.91
```

MQ cameras do not have `gain_selector` and reject this preset with `ERROR 100`; use `gain` alone there, which sets the sensor gain.

---

## Expected output

`save`:

```text
Opening camera index 0
Applying to the camera:
  exposure = 10000
  gain = 2.0
Preset saved to FFS file 'User1'
```

`show` (or no command):

```text
Opening camera index 0
Preset in FFS file 'User1':
  exposure = 10000
  gain = 2.0
```

On a camera without a preset, `show` prints `No preset stored in FFS file 'User1'` and exits with 0, while `load` reports an error.

`load`:

```text
Opening camera index 0
Applying preset from FFS file 'User1':
  exposure = 10000
  gain = 2.0
Preset applied
```

Errors go to stderr and the sample exits with 1. A misspelled parameter, an unknown enum name or a value of the wrong type is reported before the camera is opened, for example `Error: 'XI_GAIN_DIGITAL' is not a value of gain_selector; use one of: ...`. A value the camera rejects is reported as `Error: the camera rejected gain=99: ...`, and nothing is stored. Without a camera, the sample prints `Error: xiAPI call failed: ERROR 56: No Devices Found`. On a camera without FFS support, it prints `Error: this camera does not support the flash file system (FFS): ...`. If `User1` holds data that was not written by this sample, `show` and `load` refuse to use it.

---

## Known limitations / caveats

- The preset is stored in `User1` because MX, MC, CB and MT cameras accept application data only in `User1`, `User2` and `User3`, while other models accept any name; `User1` works on every family. Any other data your application keeps in `User1` is overwritten by `save`.
- The preset must be ASCII text of at most 4096 bytes.
- 64-bit integer parameters (read-only counters such as `timestamp`) and command parameters are rejected, because they do not describe a camera setting.
- The sample does not apply the preset by itself when the camera opens; your application must call `load` (or `load_preset`) after opening it.
- Available parameters and their ranges depend on the camera model; see [Camera-family-specific presets](#camera-family-specific-presets).
- Not every XIMEA camera has a flash file system. The sample uses only the basic FFS parameters (`ffs_file_name`, `ffs_file_size`, `read_file_ffs`, `write_file_ffs`) and reports a camera without FFS as an error.
- A missing preset is detected from a file size of 0 or from error 113 (`FFS file not found`). The sample checks the size before reading, because reading an empty file fails on MX cameras with `ERROR 12`. Other error codes are reported, not ignored.

### Tested camera families

| Family | Model | Status |
|--------|-------|--------|
| MQ (xiQ) | MQ022MG-CM | Tested: `save`, `show`, `load`, `-s SERIAL`, overwriting a longer preset with a shorter one, rejection of an out-of-range value (`ERROR 11`). `gain_selector` not available (`ERROR 100`). |
| MX (xiX) | MX050MG-SY-X2G2-FF (firmware 22.30, xiAPI 4.33.26) | Tested: `save`, `show`, `load`, `-s SERIAL`, overwriting a longer preset with a shorter one, rejection of an out-of-range value (`ERROR 11`), `show` and `load` with an empty file. `gain_selector` accepts `ALL`, `ANALOG_ALL` and `DIGITAL_ALL`, each 0–24 dB; `load` sets the digital gain and leaves the analog gain unchanged. |
| MC (xiC) | — | TODO: test. |
| CB (xiB) | — | TODO: test. |
| MT (xiT) | — | TODO: test. |

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [How to work with the camera FFS](https://www.ximea.com/support/wiki/apis/How_to_work_with_FFS)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
