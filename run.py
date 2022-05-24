import subprocess




subprocess.run(["python3", "yolov5/yolov5/train.py", "--data", "children.yaml", "--weights", "yolov5s.pt" ,"--batch" , "10" , "--epochs", "50" , "--workers", "32",  "--optimizer", "Adam"])