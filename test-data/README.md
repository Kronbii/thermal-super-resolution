# test-data/

17 thermal frames, 320x256, 8-bit grayscale PNG. They are the benchmark's
built-in test set (`repo17` in `benchmark/protocol.py`) and the input of the
threaded real-time demo scripts.

## Where they come from

- **Dataset:** Teledyne FLIR ADAS v2 (thermal, FLIR Tau 2 camera, native
  640x512). The file names follow that dataset's naming,
  `video-<video id>-frame-<frame number>-<image id>`, from three source videos.
- **Processing:** 320x256 is exactly 640x512 / 2, and the demo scripts that use
  them read `datasets/flir_thermal_x2/val/LR_bicubic/X2`. So these are the x2
  low-resolution versions (OpenCV bicubic) of frames from this repository's own
  validation split, made by `scripts/prepare-train-data.ipynb`. That split was
  drawn from FLIR's `images_thermal_train` and `images_thermal_val` folders
  pooled together. It was held out from training, but it was used to pick the
  best checkpoint.
- **What this means for the benchmark:** when the benchmark treats these frames
  as the high-resolution reference, the reference is already a downscaled
  image, not a native sensor frame. Results on this set describe 320x256
  content.

This provenance comes from the file names, the image sizes and the scripts in
this repository. The original dataset files were not opened to confirm it.

## Licence

FLIR ADAS v2 is distributed under the **Teledyne FLIR Non-Commercial Image
Dataset License Agreement**: Teledyne FLIR keeps ownership, and use is limited
to research and non-commercial development. **These frames are not covered by
this repository's MIT licence.** The full text of FLIR's agreement could not be
retrieved while this note was written, so it is not settled whether
redistributing derived frames in a public repository is allowed. Check the
agreement before reusing them, and consider replacing them with frames whose
licence clearly permits redistribution.

## Integrity

`benchmark/manifests/repo17.json` lists the SHA-256 of every frame. The
benchmark refuses to run if any frame is missing or has changed.
