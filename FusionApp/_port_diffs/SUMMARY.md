# Focused clock-alignment port notes
# A = E:\Twowheelers_18_5\RoadSafetyOnTwoWheelers-main\FusionApp
# B = D:\Github\RoadSafetyOnTwoWheelers\FusionApp
# Diff direction: B -> A (what A adds for alignment)

## Line counts
| file | A | B |
|------|---|---|
| camera/d455.py | 482 | 266 |
| radar/dca1000_awr2243.py | 1303 | 1236 |
| engine/fusion_engine.py | 784 | 655 |

## recording package
- A has: recording/{__init__,clock,sync_recording,detections_csv}.py
- B has NO recording/ directory on disk
- BUT B already imports recording from vod_conversion:
  - from recording.detections_csv import save_expected_detections_csv
  - from recording.sync_recording import RecordingManifest / SESSION_FILENAME
- Port recording/ package first (prerequisite), then wire feeds/engine

## B imports from recording?
- camera/d455.py: NO
- radar/dca1000_awr2243.py: NO
- engine/fusion_engine.py: NO
- vod_conversion/*: YES (broken without package)

## Key A symbols to port

### camera/d455.py (A)
Imports:
  capture_clock_ns, calibrate_realsense_offset, realsense_ms_to_capture_ns
  CLOCK_DOMAIN
  RecordingManifest, RecordingPairState, parse_start_recording_command, relative_timestamp_s
  generate_radar_filename (for manifest)

D455Config fields: dest_dir, timestamp_origin, recording_pair_meta

State:
  _recording_pair_state, _recording_manifest, _recording_epoch_ns
  _last_pair_request_gen, _frame_ring, _rs_mono_offset_ns, _pair_lock

Methods:
  begin_recording via RecordingPairState.begin_recording(epoch)
  end_recording
  _process_pair_save_requests / read_pair_request
  _save_camera_frame (paired)
  calibrate_realsense_offset + realsense_ms_to_capture_ns
  publish_camera_frame
  capture_clock_ns for shared origin fallback

### radar/dca1000_awr2243.py (A) — live DCA1000EVM only
Imports: capture_clock_ns, RecordingPairState, parse_start_recording_command, relative_timestamp_s

DCA1000Config: +timestamp_origin; dest_dir Optional + CFGS.new_recording_dir()
DCA1000EVM.__init__: +recording_pair_meta
State: _recording_pair_state, _recording_epoch_ns
request_pair_save(capture_mono_ns) on save
begin_recording / end_recording on control cmds
run(): RecordingPairState.attach; start_time from timestamp_origin or capture_clock_ns

Also A-only radar live extras (related but not pure clock):
  incomplete-frame drop (return None)
  src_filepath in SHM meta
  block queue while recording

### engine/fusion_engine.py (A)
from recording.clock import capture_clock_ns
_create_radar_feed: pass recording_pair_meta
_create_camera_feed: set camera_config.recording_pair_meta
After prealloc_shm: create_recording_pair_shm + inject into both feed configs
After create feeds: shared_timestamp_origin via capture_clock_ns
Control loop: on start_recording -> begin_recording + write_recording_session
               on stop_recording -> end_recording
Cleanup: unlink _recording_pair_shm

A also has analyser kwargs B lacks (preserve B vs merge carefully):
  intensity_mode, pc_bin_intensity_mode
