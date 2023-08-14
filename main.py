import json
import torch
import numpy as np
import cv2
from models import cpm_bn_hand
import thirdparty.mytools as mytools
import thirdparty.mytransform as mytransform
import socket
import argparse
import copy
import time


def get_pdprams(pdparms_path, model):

    pdparams = {}
    with open(pdparms_path, "r") as fp:
        pdparams = dict(json.load(fp))

    with torch.no_grad():
        for name, param in model.named_parameters():
            # 跳过不需要加载的参数
            if name not in pdparams:
                continue
            # 将权重从字典 pdparams 中加载到模型中
            param.copy_(torch.from_numpy(np.array(pdparams[name])))

def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cpm_bn_hand.CPM_hand_torch(21).to(device)
    pdprams_path = "./bn_500.json"
    get_pdprams(pdprams_path, model)
    return model, device

def process_frame(frame, centermap, model, device):

    centermap = mytools.center_map_default(368, 368, 3)
    # centermap = copy.deepcopy(mytransform.to_tensor(centermap))
    centermap = torch.from_numpy(centermap).unsqueeze(0)

    img = cv2.resize(frame, (368, 368))
    img = mytransform.normalize(mytransform.to_tensor(img), np.array([128, 128, 128]),
                                np.array([256, 256, 256]))
    
    # img = np.transpose(img, (2, 0 ,1))
    # img = img.astype(np.float32)
    img = img.unsqueeze(0)

    with torch.no_grad():
        reason_start_time = time.time()  # 记录开始时间
        heat1, heat2, heat3, heat4, heat5, heat6 = model(img.to(device), centermap.to(device))
        reaon_end_time = time.time()  # 记录结束时间

        reason_cost_time = reaon_end_time - reason_start_time  # 计算耗时，单位为秒
        print("reason_cost_time:", reason_cost_time * 1000., "ms")
        heatmapi = heat6[0].cpu().numpy()
        kptsi = mytools.get_kpts_from_heatmap(heatmapi, 368., 368.)
        img0 = np.transpose(img[0], (1, 2, 0)).numpy()
        img0 = img0 * np.array([256.,256., 256.], dtype=np.float32) + np.array([128., 128., 128.], dtype=np.float32)
        cv2.imwrite("debug.jpg", img0)
        imagei_p = mytools.draw_paint(img0.copy(), kptsi)
        cv2.imwrite("debug1.jpg", imagei_p)

    return imagei_p


def main(args):

    model, device = load_model()

    # 创建TCP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # WSL虚拟机的IP地址和端口号
    wsl_address = (args.wsl_ip, args.port)
    print(f"ip={args.wsl_ip}, port={args.port}")

    try:
        sock.bind(wsl_address)
        print("connect success.")
        BUFFER_SIZE = 655070  # 数据包大小限制

        # 设置了以图像中心为gauss核的centermap
        centermap = mytools.center_map_default(368, 368, 3)
        # centermap = torch.from_numpy(centermap)
        # centermap = centermap.unsqueeze(0) # 添加一个维度，变成1 * 1 * H * W

        count = 0

        while True:
            frame_start_time = time.time()  # 记录开始时间

            # 接收数据
            data, _ = sock.recvfrom(BUFFER_SIZE)
            # print("Received:", len(data), "bytes")
            try:
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow('frame', frame)
                imagei_p = process_frame(frame, centermap, model, device)
                print(imagei_p.shape)
                # 在WSL中显示图像
                cv2.imshow('process img', imagei_p)
                if cv2.waitKey(1) == ord('q'):
                    break

                frame_end_time = time.time()  # 记录结束时间

                frame_cost_time = frame_end_time - frame_start_time  # 计算耗时，单位为秒
                print("frame_cost_time:", frame_cost_time * 1000., "ms")
            except Exception as e:
                print("Error decoding image:", e)

            count += 1
            if (count % 100) == 0:
                cv2.imwrite(f'./output/{count}.jpg', imagei_p)

    except Exception as e:
        print("Error:", e)

    finally:
        # 关闭套接字和窗口
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('wsl_ip', type=str, nargs='?', default='172.21.246.1', help='WSL IP')
    parser.add_argument('port', type=int, nargs='?', default='8888', help='port=8888')
    args = parser.parse_args()

    main(args)