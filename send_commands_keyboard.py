#!/usr/bin/env python3
"""
Keyboard velocity command publisher for Unitree wholebody control.
"""

import argparse
import select
import sys
import termios
import time
import tty
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

import threading
import math
import numpy as np
import time

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None


class LowPassFilter:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._value = 0.0
        self._last_value = 0.0

    def update(self, new_value, max_accel=1.5):
        delta = new_value - self._last_value
        delta = np.clip(delta, -max_accel, max_accel)
        filtered = self.alpha * (self._last_value + delta) + (1 - self.alpha) * self._value
        self._last_value = filtered
        self._value = filtered
        return self._value


class KeyboardController:
    def __init__(self, mode="auto"):
        self.control_params = {
            'x_vel': 0.0,
            'y_vel': 0.0,
            'yaw_vel': 0.0,
            'height': 0.0
        }
        
        # Key increment step size   
        self.increment = 0.05
        
        # control range
        self.ranges = {
            'x_vel': (-0.6, 1.0),    # forward velocity
            'y_vel': (-0.5, 0.5),   # lateral velocity
            'yaw_vel': (-1.57, 1.57), # yaw velocity
            'height': (-0.5, 0.0)    # height
        }
        
        # key state
        self.key_states = {
            'w': False,  # forward
            's': False,  # backward
            'a': False,  # left
            'd': False,  # right
            'z': False,  # left rotation
            'x': False,  # right rotation
            'c': False,  # crouch
        }
        self._terminal_key_times = {key: 0.0 for key in self.key_states}
        self._terminal_key_hold_timeout = 0.25
        
        self.param_lock = threading.Lock()
        self.running = True
        self.input_mode = self._select_input_mode(mode)

        self._filters = {
            'x_vel': LowPassFilter(alpha=0.3),
            'y_vel': LowPassFilter(alpha=0.3),
            'yaw_vel': LowPassFilter(alpha=0.3),
            'height': LowPassFilter(alpha=0.3)
        }

        self._default_values = {
            'x_vel': 0.0,
            'y_vel': 0.0,
            'yaw_vel': 0.0,
            'height': 0.0
        }

        # Start threads
        self._control_thread = threading.Thread(target=self._control_update)
        self._control_thread.daemon = True
        self._control_thread.start()

        # Start keyboard listener
        self._start_keyboard_listener()

    def _select_input_mode(self, mode):
        """Select terminal input when possible because it works over SSH/terminal sessions."""
        if mode == "auto":
            if sys.stdin.isatty():
                return "terminal"
            if pynput_keyboard is not None:
                return "pynput"
            raise RuntimeError("stdin is not a TTY and pynput is not installed")
        if mode == "terminal" and not sys.stdin.isatty():
            raise RuntimeError("terminal input mode requires an interactive TTY")
        if mode == "pynput" and pynput_keyboard is None:
            raise RuntimeError("pynput input mode requested, but pynput is not installed")
        return mode

    def _start_keyboard_listener(self):
        """start selected keyboard listener"""
        if self.input_mode == "terminal":
            self._start_terminal_listener()
        elif self.input_mode == "pynput":
            self._start_pynput_listener()
        else:
            raise RuntimeError(f"unsupported keyboard input mode: {self.input_mode}")

    def _start_pynput_listener(self):
        """start pynput keyboard listener"""
        def on_press(key):
            """key press event"""
            try:
                key_char = key.char.lower() if hasattr(key, 'char') and key.char else None
                
                with self.param_lock:
                    if key_char in self.key_states:
                        if not self.key_states[key_char]:
                            self.key_states[key_char] = True
                            print(f"[KEY] {key_char.upper()}: press")
                    elif key_char == 'q':
                        print("exit program...")
                        self.running = False
                        return False  # stop listening
                        
            except AttributeError:
                # handle special keys
                pass

        def on_release(key):
            """按键释放事件"""
            try:
                key_char = key.char.lower() if hasattr(key, 'char') and key.char else None
                
                with self.param_lock:
                    if key_char in self.key_states:
                        if self.key_states[key_char]:
                            self.key_states[key_char] = False
                            print(f"[KEY] {key_char.upper()}: release")
                            
            except AttributeError:
                # handle special keys
                pass

        # start keyboard listener
        self.listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.listener.start()
        
        print("keyboard listener started in pynput mode...")
        print("press W/A/S/D/Z/X/C keys to control")
        print("press Q key to exit program")

    def _start_terminal_listener(self):
        """start terminal raw keyboard listener"""
        self._terminal_thread = threading.Thread(target=self._terminal_read_loop)
        self._terminal_thread.daemon = True
        self._terminal_thread.start()

        print("keyboard listener started in terminal mode...")
        print("focus this terminal and press W/A/S/D/Z/X/C directly; do not press Enter")
        print("press Q key to exit program")

    def _terminal_read_loop(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        self._terminal_old_settings = old_settings
        try:
            tty.setcbreak(fd)
            new_settings = termios.tcgetattr(fd)
            new_settings[3] = new_settings[3] & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

            while self.running:
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not readable:
                    continue
                key_char = sys.stdin.read(1).lower()
                if not key_char:
                    continue

                with self.param_lock:
                    if key_char in self.key_states:
                        if not self.key_states[key_char]:
                            print(f"[KEY] {key_char.upper()}: press")
                        self.key_states[key_char] = True
                        self._terminal_key_times[key_char] = time.monotonic()
                    elif key_char == 'q':
                        print("exit program...")
                        self.running = False
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _expire_terminal_keys_locked(self):
        if self.input_mode != "terminal":
            return
        now = time.monotonic()
        for key_char, last_seen in self._terminal_key_times.items():
            if self.key_states[key_char] and now - last_seen > self._terminal_key_hold_timeout:
                self.key_states[key_char] = False

    def _control_update(self):
        """control parameter update thread"""
        while self.running:
            with self.param_lock:
                self._expire_terminal_keys_locked()

                # update control parameters according to key states
                
                # forward/backward (x_vel)
                if self.key_states['w']:  # forward
                    self.control_params['x_vel'] = min(
                        self.control_params['x_vel'] + self.increment,
                        self.ranges['x_vel'][1]
                    )
                elif self.key_states['s']:  # backward
                    self.control_params['x_vel'] = max(
                        self.control_params['x_vel'] - self.increment,
                        self.ranges['x_vel'][0]
                    )
                else:
                    # release key, gradually return to default value
                    if self.control_params['x_vel'] > 0:
                        self.control_params['x_vel'] = max(0, self.control_params['x_vel'] - self.increment * 2)
                    elif self.control_params['x_vel'] < 0:
                        self.control_params['x_vel'] = min(0, self.control_params['x_vel'] + self.increment * 2)

                # left/right (y_vel)
                if self.key_states['a']:  # left
                    self.control_params['y_vel'] = max(
                        self.control_params['y_vel'] - self.increment,
                        self.ranges['y_vel'][0]
                    )
                elif self.key_states['d']:  # right
                    self.control_params['y_vel'] = min(
                        self.control_params['y_vel'] + self.increment,
                        self.ranges['y_vel'][1]
                    )
                else:
                    # release key, gradually return to default value
                    if self.control_params['y_vel'] > 0:
                        self.control_params['y_vel'] = max(0, self.control_params['y_vel'] - self.increment * 2)
                    elif self.control_params['y_vel'] < 0:
                        self.control_params['y_vel'] = min(0, self.control_params['y_vel'] + self.increment * 2)

                # left/right rotation (yaw_vel)
                if self.key_states['z']:  # left
                    self.control_params['yaw_vel'] = max(
                        self.control_params['yaw_vel'] - self.increment,
                        self.ranges['yaw_vel'][0]
                    )
                elif self.key_states['x']:  # right
                    self.control_params['yaw_vel'] = min(
                        self.control_params['yaw_vel'] + self.increment,
                        self.ranges['yaw_vel'][1]
                    )
                else:
                    # release key, gradually return to default value
                    if self.control_params['yaw_vel'] > 0:
                        self.control_params['yaw_vel'] = max(0, self.control_params['yaw_vel'] - self.increment * 2)
                    elif self.control_params['yaw_vel'] < 0:
                        self.control_params['yaw_vel'] = min(0, self.control_params['yaw_vel'] + self.increment * 2)

                # crouch (height)
                if self.key_states['c']:  # crouch
                    self.control_params['height'] = max(
                        self.control_params['height'] - self.increment,
                        self.ranges['height'][0]
                    )
                else:
                    # release key, gradually return to default value
                    if self.control_params['height'] < 0:
                        self.control_params['height'] = min(0, self.control_params['height'] + self.increment * 2)

                # round to avoid floating point precision issues
                for key in self.control_params:
                    self.control_params[key] = round(self.control_params[key], 3)

            time.sleep(0.02)  # 50Hz update frequency

    # === external interface ===

    def get_control_params(self):
        with self.param_lock:
            return self.control_params.copy()

    def get_key_states(self):
        with self.param_lock:
            return self.key_states.copy()
    
    def stop(self):
        """stop keyboard controller"""
        self.running = False
        if hasattr(self, 'listener'):
            self.listener.stop()
        if hasattr(self, '_terminal_old_settings') and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._terminal_old_settings)
            except termios.error:
                pass


def publish_reset_category(category, publisher):
    # construct message
    msg = String_(data=str(category))  # pass data parameter directly during initialization

    # create publisher

    # publish message
    publisher.Write(msg)
    # print(f"published reset category: {category}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish wholebody run commands from keyboard input.")
    parser.add_argument(
        "--mode",
        choices=["auto", "terminal", "pynput"],
        default="auto",
        help="keyboard input mode. auto uses terminal mode when stdin is a TTY.",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("keyboard control instructions:")
    print("W: forward    S: backward")
    print("A: left  D: right") 
    print("Z: left rotation  X: right rotation")
    print("C: crouch    Q: exit program")
    print("terminal mode: press keys directly, no Enter")
    print("pynput mode: press and hold keys in the active desktop session")
    print("")
    print("requested input mode:", args.mode)
    print("=" * 50)
    
    try:
        # initialize DDS
        print("initializing DDS communication...")
        ChannelFactoryInitialize(1)
        publisher = ChannelPublisher("rt/run_command/cmd", String_)
        publisher.Init()
        print("DDS communication initialized")
        
        print("initializing keyboard controller...")
        keyboard_controller = KeyboardController(mode=args.mode)
        default_height = 0.8
        
        print("=" * 50)
        print("program started, waiting for keyboard input...")
        print("press Ctrl+C to exit program")
        print("=" * 50)
        
        # add a counter, only show when the command changes
        counter = 0
        last_commands = [0.0, 0.0, 0.0, 0.8]
        
        while keyboard_controller.running:
            time.sleep(0.01)
            commands = keyboard_controller.get_control_params()
            commands['height'] = default_height + commands['height']
            
            # convert to list format string [x_vel, y_vel, yaw_vel, height]
            commands_list = [float(commands['x_vel']), -float(commands['y_vel']), -float(commands['yaw_vel']), float(commands['height'])]
            commands_str = str(commands_list)
            
            # only show when the command changes
            counter += 1
            if commands_list != last_commands:
                print(f"commands: {commands_str}")
                last_commands = commands_list.copy()
                
            publish_reset_category(commands_str, publisher)
            
    except KeyboardInterrupt:
        print("\nprogram interrupted by user (Ctrl+C)")
        if 'keyboard_controller' in locals():
            keyboard_controller.stop()
    except Exception as e:
        print(f"\nprogram error: {e}")
        if 'keyboard_controller' in locals():
            keyboard_controller.stop()
    
    print("program ended") 
