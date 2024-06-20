trtexec --onnx=sim_os_track.onnx --saveEngine=sim_os_track_fp32.engine  --verbose  --inputShape=x:1x3x256x256 --inputShape=z:1x3x128x128 --workspace=1400
