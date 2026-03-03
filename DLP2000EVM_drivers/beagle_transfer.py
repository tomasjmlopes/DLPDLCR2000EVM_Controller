#!/usr/bin/env python3
"""
beagle_transfer.py

A class-based client to send BMP images to a BeagleBone Black,
maintain a persistent SSH/SFTP connection, flush remote images,
display them fullscreen via feh, close the display,
generate/send predefined masks at 640×360 resolution,
and provide a stop() to display a full black mask.
"""
import os
import time
import tempfile
from typing import Optional, List

import struct
import paramiko
from PIL import Image
import numpy as np


class BeagleBoneImageClient:
    """
    Encapsulates sending, flushing, displaying, closing BMP images,
    and generating/sending predefined masks on a BeagleBone Black
    over a persistent SSH/SFTP connection.
    """
    def __init__(
        self,
        host: str = "192.168.7.2",
        username: str = "debian",
        password: str = "temppwd",
        remote_path: str = "/home/debian/temp_images/",
        width: int = 640,
        height: int = 360,
        pitch_um: float = 7.56,
    ):
        self.host = host
        self.username = username
        self.password = password
        self.remote_path = remote_path.rstrip("/") + "/"
        # DMD properties
        self.width = width
        self.height = height
        self.pitch_um = pitch_um

        # Persistent connections
        self.ssh: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None
        self._connect()

    def _warp_cursor(self, x: int = 0, y: int = 0) -> None:
        """
        Warp the BBB’s X mouse to (x,y) immediately after connecting.
        """
        # ensure DISPLAY is set, and run in background so it doesn't block
        cmd = f"DISPLAY=:0 /home/{self.username}/move_cursor {x} {y} &"
        self.ssh.exec_command(cmd)


    def _connect(self):
        """
        Establish or refresh persistent SSH and SFTP connections.
        """
        if not self.ssh or not self.ssh.get_transport() or not self.ssh.get_transport().is_active():
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # self.ssh.connect(self.host, username=self.username, password=self.password)
            self.ssh.connect(self.host, username=self.username, password=self.password, compress=True)
            self.sftp = self.ssh.open_sftp()
            self._warp_cursor(0, 0)

    def close_connection(self) -> None:
        """
        Close persistent SSH and SFTP connections.
        """
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        if self.ssh:
            self.ssh.close()
            self.ssh = None

    def send_image(self, image_array: np.ndarray) -> str:
        """
        Send a BMP image array of shape (height, width) or (height, width, 3) to the BBB.

        Returns the remote filepath of the uploaded image.
        """
        self._connect()
        # Validate
        if not isinstance(image_array, np.ndarray):
            raise TypeError("Input must be a numpy ndarray.")
        if image_array.dtype != np.uint8:
            raise ValueError(f"Image dtype must be uint8, got {image_array.dtype}")
        # Check dimensions
        dims = image_array.shape
        if image_array.ndim == 3:
            h, w, _ = dims
        elif image_array.ndim == 2:
            h, w = dims
        else:
            raise ValueError(f"Invalid image dimensions: {dims}")
        if (w, h) != (self.width, self.height):
            raise ValueError(f"Image must be exactly {self.width}x{self.height} (got {w}x{h})")

        # Save to temp BMP
        with tempfile.NamedTemporaryFile(prefix="bbb_img_", suffix=".bmp", delete=False) as tmp:
            temp_path = tmp.name
        Image.fromarray(image_array).save(temp_path, format='BMP')

        # Upload
        try:
            try:
                self.sftp.stat(self.remote_path)
            except IOError:
                self.sftp.mkdir(self.remote_path)
            filename = os.path.basename(temp_path)
            remote_file = f"{self.remote_path}{filename}"
            self.sftp.put(temp_path, remote_file)
        finally:
            os.remove(temp_path)
        return remote_file

    def generate_mask(self, mask_type: str, radius_um: Optional[float] = None) -> np.ndarray:
        """
        Create a mask array at the DMD resolution (640×360):
        - "white": all pixels = 255
        - "black": all pixels = 0
        - "grid": 32×32 px squares alternating black/white
        - "circle": white circle on black with radius in microns
        """
        w, h = self.width, self.height
        if mask_type == "white":
            return np.full((h, w), 255, dtype=np.uint8)
        if mask_type == "black":
            return np.zeros((h, w), dtype=np.uint8)
        if mask_type == "grid":
            arr = np.zeros((h, w), dtype=np.uint8)
            cell = 32
            for y in range(0, h, cell):
                for x in range(0, w, cell):
                    if ((x // cell) + (y // cell)) % 2 == 0:
                        arr[y:y+cell, x:x+cell] = 255
            return arr
        if mask_type == "circle":
            if radius_um is None:
                raise ValueError("radius_um must be provided for circle mask.")
            r_px = int(radius_um / self.pitch_um)
            cy, cx = h // 2, w // 2
            yy, xx = np.ogrid[:h, :w]
            mask = ((yy - cy)**2 + (xx - cx)**2) <= (r_px**2)
            arr = np.zeros((h, w), dtype=np.uint8)
            arr[mask] = 255
            return arr
        raise ValueError(f"Unknown mask_type '{mask_type}'")

    def preset_mask(self, mask_type: str, radius_um: Optional[float] = None) -> str:
        """
        Generate and send a predefined mask, returning its remote path.
        """
        mask = self.generate_mask(mask_type, radius_um)
        self.send_image(mask)
        self.show_image()


    def flush_remote_images(self) -> None:
        """
        Remove all BMP files in the remote directory.
        """
        self._connect()
        cmd = f"rm -f {self.remote_path}*.bmp"
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            err = stderr.read().decode().strip()
            raise RuntimeError(f"Failed to flush remote images: {err}")

    def show_image(self, remote_file: Optional[str] = None) -> None:
        """
        Close any existing image then display the specified (or latest) BMP fullscreen via feh.
        """
        # Ensure only one instance
        # self.close_image()
        self._connect()
        # Determine target file
        if remote_file is None:
            files = self._get_latest_remote_list()
            if not files:
                raise FileNotFoundError("No BMP images found on the BBB.")
            remote_file = files[0]
        # Launch
        cmd = f"export DISPLAY=:0; feh -F {remote_file} &"
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        stdout.channel.recv_exit_status()
    
    def send_sequence_blob(self, masks, remote_file="/home/debian/sequence.bin", flush=True):
        self._connect()

        # packed 1bpp: 640*360/8 = 28800 bytes per frame
        frame_bytes = (self.width * self.height) // 8
        if (self.width % 8) != 0:
            raise ValueError("Width must be multiple of 8 for packed format")

        if flush:
            self.ssh.exec_command(f"rm -f {remote_file}")

        rf = self.sftp.open(remote_file, "wb")
        try:
            rf.set_pipelined(True)

            for i, m in enumerate(masks):
                if m.dtype != np.uint8 or m.shape != (self.height, self.width):
                    raise ValueError(f"Mask {i} must be uint8 with shape {(self.height, self.width)}")

                bits = (m > 0).astype(np.uint8)
                packed = np.packbits(bits, axis=1)
                raw = packed.tobytes()
                if len(raw) != frame_bytes:
                    raise RuntimeError(f"Bad packed frame size: got {len(raw)}, expected {frame_bytes}")

                rf.write(raw)

            rf.flush()
        finally:
            rf.close()

        return remote_file
        
    def play_sync_out(self, remote_file, frames, gpio_in, gpio_out, loop=1, warmup_edges=2, timeout_ms=2000, pulse_each=1):
        self._connect()

        ### Refresh display to recover the vsync pulse
        self.start_desktop()
        time.sleep(0.1)
        self.stop_desktop()
        time.sleep(0.1)

        cmd = (
            "sudo -n taskset -c 0 /home/{u}/fb_player_syncout_rt "
            "--file {f} --frames {n} --loop {loop} "
            "--gpio_in {gin} --gpio_out {gout} "
            "--warmup_edges {w} --timeout_ms {t} --pulse_each {p} "
            "--rt_prio 80 --expect_hz 120 --report_every 0"
        ).format(
            u=self.username,
            f=remote_file,
            n=int(frames),
            loop=int(loop),
            gin=int(gpio_in),
            gout=int(gpio_out),
            w=int(warmup_edges),
            t=int(timeout_ms),
            p=int(pulse_each),
        )
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        code = stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="ignore")
        err = stderr.read().decode(errors="ignore")

        self.stop_desktop()
        if code != 0:
            raise RuntimeError("GPIO sync/out player failed ({}).\nSTDOUT:\n{}\nSTDERR:\n{}".format(code, out, err))
        
    def stop_desktop(self) -> None:
        """
        Stop the display manager so nothing else draws to /dev/fb0.
        Safe and reversible. Does not affect SSH.
        """
        self._connect()
        cmd = "sudo systemctl stop display-manager"
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            err = stderr.read().decode(errors="ignore")
            raise RuntimeError("Failed to stop desktop:\n{}".format(err))

    def start_desktop(self) -> None:
        """
        Restart the display manager after framebuffer experiments.
        """
        self._connect()
        cmd = "sudo systemctl start display-manager"
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            err = stderr.read().decode(errors="ignore")
            raise RuntimeError("Failed to start desktop:\n{}".format(err))
    
    def close_image(self) -> None:
        """
        Kill any running feh process to close fullscreen display.
        """
        self._connect()
        cmd = "pkill feh"
        # ignore errors
        self.ssh.exec_command(cmd)

    def stop(self) -> None:
        """
        Send and display a full black mask (equivalent to "stop" display).
        """
        # flush previous display and show black mask
        self.close_image()
        black_path = self.preset_mask("black")
        self.show_image(black_path)

    def _get_latest_remote_list(self) -> List[str]:
        """
        Return remote BMPs sorted by modification time, newest first.
        """
        self._connect()
        cmd = f"ls -t {self.remote_path}*.bmp"
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        return stdout.read().decode().split()
