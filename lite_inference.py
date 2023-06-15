import numpy as np
import cv2
import matplotlib.cm as cm
import onnxruntime as ort
import jetson_camera
import yolo_nms

cv2.namedWindow('out', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('out',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
session = ort.InferenceSession('yolov8m-pose.onnx', providers=[('TensorrtExecutionProvider', {'trt_engine_cache_enable': True, 'trt_engine_cache_path': '/home/nvidia/TRT_cache/engine_cache', "trt_fp16_enable": True, 'device_id': 0, }), 'CUDAExecutionProvider']) # providers=['CPUExecutionProvider'])#,

input_name = session.get_inputs()[0].name

def model_inference(input=None):
    output = session.run([], {input_name: input})
    return output[0]

IMG_SZ=(736,480)

sk = [15,13, 13,11, 16,14, 14,12, 11,12, 
            5,11, 6,12, 5,6, 5,7, 6,8, 7,9, 8,10, 
            1,2, 0,1, 0,2, 1,3, 2,4, 3,5, 4,6]

def preprocess_img(frame):
    img = frame[:, :, ::-1]
    img = img/255.00
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img


def single_non_max_suppression(prediction):
    argmax = np.argmax(prediction[4,:])
    x = (prediction.T)[argmax]
    
    box = x[:4] #Cx,Cy,w,h
    conf = x[4]
    keypts = x[5:]

    return box, conf, keypts

def post_process_multi(img, output, score_threshold=10):
    boxes, conf_scores, keypt_vectors = yolo_nms.non_max_suppression(output, score_threshold)

    for keypts, conf in zip(keypt_vectors, conf_scores):
        plot_keypoints(img, keypts, score_threshold)
    return img

def post_process_single(img, output, score_threshold=10):
    box, conf, keypts = single_non_max_suppression(output)
    keypts = smooth_pred(keypts)
    plot_keypoints(img, keypts, score_threshold)
    return img



def plot_keypoints(img, keypoints, threshold=10):
    for i in range(0,len(sk)//2):
        pos1 = (int(keypoints[3*sk[2*i]]), int(keypoints[3*sk[2*i]+1]))
        pos2 = (int(keypoints[3*sk[2*i+1]]), int(keypoints[3*sk[2*i+1]+1]))
        conf1 = keypoints[3*sk[2*i]+2]
        conf2 = keypoints[3*sk[2*i+1]+2]

        color = (cm.jet(i/(len(sk)//2))[:3])
        color = [int(c * 255) for c in color[::-1]]
        if conf1>threshold and conf2>threshold: # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(img, pos1, pos2, color, thickness=8)

    for i in range(0,len(keypoints)//3):
        x = int(keypoints[3*i])
        y = int(keypoints[3*i+1])
        conf = keypoints[3*i+2]
        if conf > threshold: # Only draw the circle if confidence is above some threshold
            cv2.circle(img, (x, y), 3, (0,0,0), -1)

keypoints_old = None
def smooth_pred(keypoints):
    global keypoints_old
    if keypoints_old is None:
        keypoints_old = keypoints.copy()
        return keypoints
    
    smoothed_keypoints = []
    for i in range(0, len(keypoints), 3):
        x_keypoint = keypoints[i]
        y_keypoint = keypoints[i+1]
        conf = keypoints[i+2]
        x_keypoint_old = keypoints_old[i]
        y_keypoint_old = keypoints_old[i+1]
        conf_old = keypoints_old[i+2]
        x_smoothed = (conf * x_keypoint + conf_old * x_keypoint_old)/(conf+conf_old)
        y_smoothed = (conf * y_keypoint + conf_old * y_keypoint_old)/(conf+conf_old)
        smoothed_keypoints.extend([x_smoothed, y_smoothed, (conf+conf_old)/2])
    keypoints_old = smoothed_keypoints
    return smoothed_keypoints

if __name__== "__main__":
    cap = jetson_camera.VideoCapture(out_width=736, out_height=480)
    while True:
        frame = cap.read()
        input_img = preprocess_img(frame)
        output = model_inference(input_img)
        # frame = post_process_single(frame, output[0], score_threshold=5)
        frame = post_process_multi(frame, output[0], score_threshold=0.2)
        
        cv2.imshow('out',frame)
        cv2.waitKey(1)