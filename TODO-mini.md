
Build a robust, reliable standalone python script that detects patterns in video streams using GStreamer, yolo and Deepstream. The script should be able to handle custom models, and provide a way to configure the detection parameters for each pattern.

The script must be robust to errors, with automatic recovery from failures. It should also provide logging and monitoring capabilities to track the performance and status of the pattern detection.

The script must be designed to work with multiple video sources (rtsp, webrtc, file, etc.) simultaneously and support real-time processing. It should be able to handle different video formats and resolutions, and provide a way to configure the detection parameters for each pattern. It should also provide a way to register and unregister (and enable or disable) video sources dynamically, e.g. by POST'ing to a REST API.

The script is a standalone application that can be run independently or integrated into larger systems.

The script should be designed to be efficient and performant, with low latency and high throughput. It should be able to handle high-resolution video streams and process multiple streams simultaneously without significant performance degradation.

It should use deepstream streamer elements for video processing, and provide a way to configure the pipeline dynamically. It should also provide a way to monitor the performance of the pipeline and detect any issues that may arise during processing.

