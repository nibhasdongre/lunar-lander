import gym
from gym.spaces import Box
import numpy as np
from flask import Flask, request, jsonify
import threading
import queue
import time
import subprocess
import os
import platform

STATE_SIZE = 17
ACTION_SIZE = 6
MAX_EPISODE_STEPS = 6000

# --- Constants ---
TARGET_LANDING_POS = np.array([0.0, 0.0, 0.1])
REWARD_SUCCESSFUL_LANDING = 250.0
PENALTY_CRASH = -200.0
PENALTY_TILTED_LANDING = -100.0
PENALTY_OUT_OF_BOUNDS = -150.0
LOW_ALTITUDE_THRESHOLD = 0.1
MAX_LANDING_SPEED_VERTICAL = 0.75
MAX_LANDING_SPEED_HORIZONTAL = 0.75
MAX_GENERAL_SPEED = 5.0
# --- FIX: Increased angular speed threshold to a more realistic value (in radians/sec) ---
MAX_ANGULAR_SPEED = np.pi * 2 # Allows for one full rotation per second before penalizing
ORIENTATION_PITCH_THRESHOLD = np.deg2rad(10)
ORIENTATION_ROLL_THRESHOLD = np.deg2rad(10)
TILT_CRASH_PITCH_THRESHOLD = np.deg2rad(45)
TILT_CRASH_ROLL_THRESHOLD = np.deg2rad(45)
BOUNDS_XY = 10000.0 # Using a large boundary
BOUNDS_Z_MAX = 1000.0
BOUNDS_Z_MIN_CRASH = -10.0

class UnrealLunarLanderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, flask_port=5000, host='0.0.0.0',
                 unreal_exe_path=None, launch_unreal=False, ue_launch_args=None):
        super(UnrealLunarLanderEnv, self).__init__()
        # Disable Flask/werkzeug request logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.action_space = Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        
        # Increased bounds for angular velocity to handle real values from UE
        MAX_ANG_VEL_BOUND = np.deg2rad(720) # Allow for very fast spins up to 2 rotations/sec

        low_bounds = np.array(
            [-BOUNDS_XY] * 3 + [-np.pi] * 3 + [-MAX_GENERAL_SPEED * 5] * 3 +
            [-MAX_ANG_VEL_BOUND] * 3 + [BOUNDS_Z_MIN_CRASH] * 4 + [0.0] * 1, dtype=np.float32
        )
        high_bounds = np.array(
            [BOUNDS_XY] * 3 + [np.pi] * 3 + [MAX_GENERAL_SPEED * 5] * 3 +
            [MAX_ANG_VEL_BOUND] * 3 + [BOUNDS_Z_MAX] * 4 + [1.0] * 1, dtype=np.float32
        )
        self.observation_space = Box(low=low_bounds, high=high_bounds, shape=(STATE_SIZE,), dtype=np.float32)

        self.flask_app = Flask(__name__)
        self.host = host
        self.port = flask_port
        self.server_thread = None
        self.data_from_unreal = queue.Queue(maxsize=1)
        self.action_for_unreal = queue.Queue(maxsize=1)
        self.unreal_exe_path = unreal_exe_path
        self.launch_unreal = launch_unreal
        self.ue_launch_args = ue_launch_args if ue_launch_args is not None else []
        self.unreal_process = None
        self._ue_connected=False
        self._ue_disconnected=False
        self.current_state = np.zeros(STATE_SIZE, dtype=np.float32)
        self.episode_step_count = 0
        self._setup_flask_routes()
        self._start_flask_server()
        if self.launch_unreal: self._launch_unreal_engine()
        print(f"UnrealLunarLanderEnv initialized. Flask server running on http://{self.host}:{self.port}")


    def _launch_unreal_engine(self):
        if not self.unreal_exe_path or not os.path.exists(self.unreal_exe_path): return
        try:
            cmd = [self.unreal_exe_path] + self.ue_launch_args
            self.unreal_process = subprocess.Popen(cmd)
            print(f"UE process started (PID: {self.unreal_process.pid}).")
        except Exception as e:
            self.unreal_process = None
            print(f"[Error] Failed to launch UE: {e}")

    def _setup_flask_routes(self):
        @self.flask_app.route('/control', methods=['POST'])
        def control_lander_route():
            try:
                data = request.json

                # on first successful UE→Python call
                if not self._ue_connected:
                    print("[UE↔Python] Connection established.")
                    self._ue_connected = True
                    self._ue_disconnected = False

                self.data_from_unreal.put(data)
                command_from_agent = self.action_for_unreal.get(timeout=10.0)
                return jsonify(command_from_agent)

            except Exception as e:
                # once-only on a lost/failed call
                if not self._ue_disconnected:
                    print(f"[UE↔Python] Connection lost: {e}")
                    self._ue_disconnected = True
                    self._ue_connected = False

                return jsonify({"command": "dummy", "error": str(e)}), 500

    def _start_flask_server(self):
        self.server_thread = threading.Thread(target=lambda: self.flask_app.run(host=self.host, port=self.port, debug=False, use_reloader=False))
        self.server_thread.daemon = True
        self.server_thread.start()

    def _calculate_reward_and_done(self, state):
        """
        Compute the reward and done flag for the given state.
        Includes step bonus, distance shaping, velocity shaping near ground,
        tilt and spin penalties, and terminal rewards/penalties.
        """
        # 1. Base step reward (survival bonus)
        reward = 0.1  
        done = False
        info = {}

        # 2. Unpack and scale state
        relative_pos    = state[0:3]  / 100.0    # X, Y, Z in meters
        orientation     = np.deg2rad(state[3:6]) # roll, pitch, yaw in radians
        lin_vel         = state[6:9]  / 100.0    # meters/sec
        ang_vel         = np.deg2rad(state[9:12])# radians/sec
        leg_altitudes   = state[12:16]/ 100.0    # meters
        hit_flag        = state[16]              # contact indicator
        lander_altitude = np.min(leg_altitudes)  # lowest leg height

        # 3. Distance shaping (horizontal vs vertical)
        horiz_dist = np.linalg.norm(relative_pos[:2])
        vert_dist  = abs(relative_pos[2])
        reward    -= horiz_dist * 0.01   # stronger horizontal centering
        reward    -= vert_dist  * 0.003  # encourage descent when centered

        # 4. Tilt penalty
        roll, pitch = orientation[0], orientation[1]
        is_severely_tilted = (
            abs(pitch) > TILT_CRASH_PITCH_THRESHOLD or
            abs(roll)  > TILT_CRASH_ROLL_THRESHOLD
        )
        if is_severely_tilted:
            reward -= 0.5

        # 5. Spin penalty
        angular_speed_mag = np.linalg.norm(ang_vel)
        if angular_speed_mag > MAX_ANGULAR_SPEED:
            reward -= (angular_speed_mag - MAX_ANGULAR_SPEED) * 0.1

        # 6. Slow-down shaping near ground
        vert_speed = abs(lin_vel[2])
        if lander_altitude < 5.0:
            reward -= vert_speed * 0.5

        # 7. Out-of-bounds termination
        if not done and (abs(relative_pos[0]) > BOUNDS_XY or abs(relative_pos[1]) > BOUNDS_XY):
            reward += PENALTY_OUT_OF_BOUNDS
            done = True
            info['status'] = 'out_of_bounds'

        # 8. Primary crash/landing check
        if not done and hit_flag > 0:
            is_body_hit = lander_altitude > LOW_ALTITUDE_THRESHOLD * 2
            if is_severely_tilted or is_body_hit:
                reward += PENALTY_CRASH
                info['status'] = 'body_or_tilted_crash'
            else:
                horizontal_speed = np.linalg.norm(lin_vel[:2])
                vertical_speed   = vert_speed
                is_on_target     = horiz_dist < 2.0
                is_stable_speed  = (
                    vertical_speed < MAX_LANDING_SPEED_VERTICAL and
                    horizontal_speed < MAX_LANDING_SPEED_HORIZONTAL
                )
                pitch_ok = abs(pitch) < ORIENTATION_PITCH_THRESHOLD
                roll_ok  = abs(roll)  < ORIENTATION_ROLL_THRESHOLD

                if is_on_target and is_stable_speed and pitch_ok and roll_ok:
                    reward += REWARD_SUCCESSFUL_LANDING
                    info['status'] = 'successful_landing'
                elif not is_stable_speed:
                    reward += PENALTY_CRASH
                    info['status'] = 'hard_landing_crash'
                else:
                    reward += PENALTY_TILTED_LANDING
                    info['status'] = 'safe_landing_off_target'
            done = True

        # 9. Max-steps termination
        if not done and self.episode_step_count >= MAX_EPISODE_STEPS:
            reward -= 20
            done = True
            info['status'] = 'max_steps_reached'

        # 10. Clip reward to avoid extreme values
        reward = np.clip(reward, -100, 100)

        return reward, done, info


    def reset(self):
        while not self.data_from_unreal.empty(): self.data_from_unreal.get_nowait()
        while not self.action_for_unreal.empty(): self.action_for_unreal.get_nowait()
        self.episode_step_count = 0

        if self.launch_unreal and (not self.unreal_process or self.unreal_process.poll() is not None):
            self._launch_unreal_engine()
            if not self.unreal_process:
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        
        # Using the explicit command-based protocol
        reset_command = {"command": "reset"}
        
        try:
            self.action_for_unreal.put(reset_command, timeout=2.0)
        except queue.Full:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
            
        print("Gym env: RESET. Waiting for UE to send a 'start' signal...")
        is_ready = False
        initial_state_from_ue = None
        timeout_duration = 120.0
        
        while not is_ready:
            try:
                ue_data = self.data_from_unreal.get(timeout=timeout_duration)
                
                if ue_data.get("start") is True and "state" in ue_data:
                    self.action_for_unreal.put({"command": "start_confirmed"})
                    is_ready = True
                    initial_state_from_ue = np.array(ue_data["state"], dtype=np.float32)
                    print("Gym env: 'start' signal received. Starting episode.")
                else:
                    self.action_for_unreal.put({"command": "dummy"})
            except queue.Empty:
                print(f"Timeout ({timeout_duration}s): No start signal. Initiating hard reset...")
                self.close()
                return self.reset()
            except Exception as e:
                print(f"[Env Reset Error] Handshake error: {e}. Initiating hard reset...")
                self.close()
                return self.reset()
                
        self.current_state = initial_state_from_ue
        return self.current_state, {}

    def step(self, action_payload):
        ##########
        def step(self, action_payload):
    self.episode_step_count += 1
    action_list = [float(a) for a in action_payload]
    step_command = {"command": "step", "action": action_list}

    try:
        self.action_for_unreal.put(step_command, timeout=2.0)
    except queue.Full:
        reward, _, info = self._calculate_reward_and_done(self.current_state)
        return self.current_state, reward - 50, True, False, {"error": "Action queue full", **info}
        
    try:
        ue_data = self.data_from_unreal.get(timeout=30.0)
        new_state_from_ue = np.array(ue_data["state"], dtype=np.float32)

        reward, is_terminal, info = self._calculate_reward_and_done(new_state_from_ue)
        self.current_state = new_state_from_ue
        truncated = info.get('status') == 'max_steps_reached'
        terminated = is_terminal and not truncated

        return self.current_state, reward, terminated, truncated, info

    except Exception:
        reward, _, info = self._calculate_reward_and_done(self.current_state)
        return self.current_state, reward - 50, True, False, {"error": "UE response timeout or bad data", **info}

        ##########

    def render(self, mode='human'):
        pass

    def close(self):
        if self.unreal_process and self.unreal_process.poll() is None:
            pid = self.unreal_process.pid
            print(f"Terminating UE process tree (PID: {pid})...")
            if platform.system() == "Windows":
                try: subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], check=True, capture_output=True)
                except Exception: self.unreal_process.terminate()
            else: self.unreal_process.terminate()
            try: self.unreal_process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.unreal_process.kill()
            self.unreal_process = None
