import sys
sys.path.insert(0, r'C:\\_development\\ai\\opencv-custom\\opencv\\build\\lib\\python3\\Release\\cv2.cp311-win_amd64.pyd')
import cv2

params = cv2.TrackerNano_Params()
params.backbone = './models/nanotrack_backbone_sim.onnx'
params.neckhead = './models/nanotrack_head_sim.onnx'
tracker = cv2.TrackerNano_create(params)
print(dir(tracker))