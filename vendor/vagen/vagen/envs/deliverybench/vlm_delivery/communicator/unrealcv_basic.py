# Communicator/unrealcv_basic.py
# -*- coding: utf-8 -*-

import time
import threading
import json
import numpy as np
import PIL.Image
from io import BytesIO

try:
    import cv2
except Exception:
    cv2 = None

import unrealcv


class UnrealCV(object):
    """
    Base UnrealCV wrapper.
    - Maintains a single TCP connection; all request calls are serialized using an RLock.
    - Ensures request ordering, preventing message ID conflicts.
    - PNG/BMP decoding returns numpy arrays in BGR format for consistent frontend usage.
    """

    def __init__(self, port, ip, resolution):
        self.ip = ip
        self.client = unrealcv.Client((ip, port))
        self.client.connect()

        # Wrap UnrealCV requests with a shared re-entrant lock
        self._req_lock = getattr(self, "_req_lock", None) or threading.RLock()
        self._orig_request = self.client.request
        self._orig_request_batch = getattr(self.client, "request_batch", None)

        def _locked_request(cmd, timeout=None):
            with self._req_lock:
                if timeout is None:
                    return self._orig_request(cmd)
                return self._orig_request(cmd, timeout)

        def _locked_request_batch(cmds):
            with self._req_lock:
                return self._orig_request_batch(cmds)

        self.client.request = _locked_request
        if self._orig_request_batch:
            self.client.request_batch = _locked_request_batch

        self.resolution = resolution
        self.ini_unrealcv(resolution)

    # ------------------------------------------------------
    # UnrealCV initialization
    # ------------------------------------------------------
    def ini_unrealcv(self, resolution=(320, 240)):
        """Initialize UnrealCV resolution and disable screen messages."""
        self.check_connection()
        w, h = resolution
        self.client.request(f"vrun setres {w}x{h}w", -1)
        self.client.request("DisableAllScreenMessages", -1)
        self.client.request("vrun sg.ShadowQuality 0", -1)
        self.client.request("vrun sg.TextureQuality 0", -1)
        self.client.request("vrun sg.EffectsQuality 0", -1)
        time.sleep(0.1)
        self.client.message_handler = self.message_handler

    def message_handler(self, message):
        """Optional hook for server message parsing."""
        return message

    def check_connection(self):
        """Reconnect until a stable UnrealCV connection is established."""
        while not self.client.isconnected():
            print("UnrealCV server is not running. Retrying...")
            time.sleep(1)
            self.client.connect()

    # ------------------------------------------------------
    # Actor operations
    # ------------------------------------------------------
    def spawn(self, prefab, name):
        cmd = f"vset /objects/spawn {prefab} {name}"
        self.client.request(cmd)

    def spawn_bp_asset(self, prefab_path, name):
        cmd = f"vset /objects/spawn_bp_asset {prefab_path} {name}"
        self.client.request(cmd)

    def clean_garbage(self):
        self.client.request("vset /action/clean_garbage")

    def set_location(self, loc, name):
        x, y, z = loc
        cmd = f"vset /object/{name}/location {x} {y} {z}"
        self.client.request(cmd)

    def set_orientation(self, orientation, name):
        pitch, yaw, roll = orientation
        cmd = f"vset /object/{name}/rotation {pitch} {yaw} {roll}"
        self.client.request(cmd)

    def set_scale(self, scale, name):
        x, y, z = scale
        cmd = f"vset /object/{name}/scale {x} {y} {z}"
        self.client.request(cmd)

    def set_physics(self, actor_name, has_physics: bool):
        cmd = f"vset /object/{actor_name}/physics {has_physics}"
        self.client.request(cmd)

    def set_collision(self, actor_name, has_collision: bool):
        cmd = f"vset /object/{actor_name}/collision {has_collision}"
        self.client.request(cmd)

    def set_movable(self, actor_name, is_movable: bool):
        cmd = f"vset /object/{actor_name}/object_mobility {is_movable}"
        self.client.request(cmd)

    def destroy(self, actor_name):
        cmd = f"vset /object/{actor_name}/destroy"
        self.client.request(cmd)

    # ------------------------------------------------------
    # Movement parameter APIs used by higher-level controllers
    # ------------------------------------------------------
    def set_max_speed(self, delivery_man_id, speed_cm_s: float) -> bool:
        name = self.get_delivery_man_name(delivery_man_id)
        cmd = f"vbp {name} SetMaxSpeed {float(speed_cm_s)}"
        self.client.request(cmd)

    def set_max_accel(self, delivery_man_id, accel_cm_s2: float) -> bool:
        name = self.get_delivery_man_name(delivery_man_id)
        cmd = f"vbp {name} SetMaxAccel {float(accel_cm_s2)}"
        self.client.request(cmd)

    def set_braking_decel(self, delivery_man_id, decel_cm_s2: float) -> bool:
        name = self.get_delivery_man_name(delivery_man_id)
        cmd = f"vbp {name} SetBrakingDel {float(decel_cm_s2)}"
        self.client.request(cmd)

    # ------------------------------------------------------
    # Position and orientation queries
    # ------------------------------------------------------
    def get_location(self, actor_name):
        try:
            cmd = f"vget /object/{actor_name}/location"
            res = self.client.request(cmd)
            return np.array([float(i) for i in res.split()])
        except Exception as e:
            print(f"Error in get_location: {e}")
            try:
                print("res:", res)
            except Exception:
                pass

    def get_location_batch(self, actor_names):
        cmds = [f"vget /object/{name}/location" for name in actor_names]
        results = self.client.request_batch(cmds)
        return [np.array([float(i) for i in r.split()]) for r in results]

    def get_orientation(self, actor_name):
        try:
            cmd = f"vget /object/{actor_name}/rotation"
            res = self.client.request(cmd)
            return np.array([float(i) for i in res.split()])
        except Exception as e:
            print(f"Error in get_orientation: {e}")
            try:
                print("res:", res)
            except Exception:
                pass

    def get_orientation_batch(self, actor_names):
        cmds = [f"vget /object/{name}/rotation" for name in actor_names]
        results = self.client.request_batch(cmds)
        return [np.array([float(i) for i in r.split()]) for r in results]

    # ------------------------------------------------------
    # Image acquisition
    # ------------------------------------------------------
    def read_image(self, cam_id, viewmode, mode="direct"):
        """
        Retrieve an image from a camera.
        viewmode: lit / normal / depth / object_mask
        mode:
            "direct": return PNG via memory
            "file": save PNG then load (slow)
            "fast": return BMP (faster)
        """
        if mode == "direct":
            cmd = f"vget /camera/{cam_id}/{viewmode} png"
            return self.decode_png(self.client.request(cmd))

        elif mode == "file":
            cmd = f"vget /camera/{cam_id}/{viewmode} {viewmode}{self.ip}.png"
            img_path = self.client.request(cmd)
            if cv2 is not None:
                return cv2.imread(img_path)
            pil = PIL.Image.open(img_path).convert("RGB")
            arr = np.array(pil)[:, :, ::-1]
            return arr

        elif mode == "fast":
            cmd = f"vget /camera/{cam_id}/{viewmode} bmp"
            return self.decode_bmp(self.client.request(cmd))

        return None

    def decode_png(self, res):
        """Decode PNG bytes into a BGR numpy array."""
        img = np.asarray(PIL.Image.open(BytesIO(res)))
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        return img[:, :, ::-1].copy()

    def decode_bmp(self, res):
        """Decode BMP bytes into a BGR numpy array."""
        pil = PIL.Image.open(BytesIO(res))
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil)
        return arr[:, :, ::-1].copy()