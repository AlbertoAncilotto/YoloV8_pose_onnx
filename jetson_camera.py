import cv2

class VideoCapture:
    
    def __init__(self, sensor_id=0, capture_width=1920, capture_height=1080, out_width=480, out_height=320, flip_method=0):
        self.sensor_id = sensor_id
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.flip_method = flip_method
        self.out_shape = (out_width, out_height)
        
        self.cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        
    def gstreamer_pipeline(self):
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, format=I420 ! appsink max-buffers=1 drop=true"
            % (
                self.sensor_id,
                self.capture_width,
                self.capture_height,
                self.flip_method,
            )
        )
    
    def read(self):
        _, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        frame = cv2.resize(frame, self.out_shape, interpolation=cv2.INTER_LINEAR)
        return frame
    
    def release(self):
        self.cap.release()