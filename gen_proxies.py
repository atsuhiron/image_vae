from typing import Callable
import os
import os.path
import shutil
import platform

import numpy as np
from tqdm import tqdm
from smb.SMBConnection import SMBConnection
import cv2

import config
from read_key import key


def default_filter_function(path: str) -> bool:
    dot_split = path.split(".")
    if len(dot_split) < 2:
        return False

    ext = dot_split[-1]
    if len(ext) == 0:
        return False

    if ext in ("db",):
        return False
    return True


def init_proxy_dir():
    if os.path.exists(config.PROXY_DIR):
        shutil.rmtree(config.PROXY_DIR)
    os.makedirs(config.PROXY_DIR, exist_ok=False)


def crop_and_resize(local_path: str) -> np.ndarray:
    img = cv2.imread(local_path)
    size = img.shape[0: 2]
    is_landscape = size[0] < size[1]
    asp_rat = max(size) / min(size)

    if asp_rat > config.PROXY_MAX_ASPECT_RATIO:
        if is_landscape:
            # landscape の際はportraitにする
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        long_len = img.shape[0]
        short_len = img.shape[1]
        cropped_long_len = int(short_len * config.PROXY_MAX_ASPECT_RATIO)
        if cropped_long_len % 2 == 1:
            # 奇数だと都合が悪いので、切り取る長さ、切り取られた後の長さが両方偶数になるように調整する
            cropped_long_len += 1
        edge_len = (long_len - cropped_long_len) // 2  # 切り取られる長さ

        img = img[edge_len: edge_len + cropped_long_len]
        if is_landscape:
            # landscape にもどす
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return cv2.resize(img, config.PROXY_SIZE)


def save_proxy(img: np.ndarray, ds_path: str):
    bn = os.path.basename(ds_path)
    save_path = f"{config.PROXY_DIR}/{bn}"
    cv2.imwrite(save_path, img)


class FileReceive:
    def __init__(self, connection: SMBConnection, service_name: str):
        self.conn = connection
        self.service_name = service_name

    def get_ds_file_list(self, target_dir: str, filter_func: Callable[[str], bool]) -> list[str]:
        base_names = filter(filter_func, [item.filename for item in self.conn.listPath(self.service_name, target_dir)])
        return [f"{target_dir}/{name}" for name in base_names]

    def receive_file(self, local_path_no_ext: str, ds_path: str) -> str:
        local_path = f"{local_path_no_ext}.{ds_path.split('.')[-1]}"
        with open(local_path, "wb") as f:
            self.conn.retrieveFile(self.service_name, ds_path, f)
        return local_path


if __name__ == "__main__":
    init_proxy_dir()
    conn = SMBConnection(key.ds_user_name, key.ds_password,
                         platform.uname().node, key.ds_remote_name, domain="", use_ntlm_v2=True)
    ds_file_list = None
    local_temp_path_no_ext = "temp"
    ds_target_dir = "img/3d"
    try:
        is_connected = conn.connect(key.ds_url)
        receiver = FileReceive(conn, key.ds_service_name)
        ds_file_list = receiver.get_ds_file_list(ds_target_dir, default_filter_function)

        for ds_file_path in tqdm(ds_file_list):
            local_temp_path = receiver.receive_file(local_temp_path_no_ext, ds_file_path)
            proxy_img = crop_and_resize(local_temp_path)
            save_proxy(proxy_img, ds_file_path)
            os.remove(local_temp_path)
    except Exception as e:
        print(e)
    finally:
        conn.close()
        print("closed")
