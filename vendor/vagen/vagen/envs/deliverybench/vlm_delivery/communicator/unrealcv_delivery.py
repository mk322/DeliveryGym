# communicator/unrealcv_delivery.py
# -*- coding: utf-8 -*-

import threading
from .unrealcv_basic import UnrealCV


class UnrealCvDelivery(UnrealCV):
    """
    Intermediate UnrealCV layer providing common Blueprint action wrappers
    for DeliveryMan actors. All client requests are already serialized
    in the base class to ensure thread-safe communication.
    """

    def __init__(self, port, ip, resolution):
        super().__init__(port, ip, resolution)
        # Shared lock for all BP calls to maintain thread safety
        self.lock = getattr(self, "_req_lock", None) or threading.RLock()

    # ------------------------------------------------------
    # Blueprint action wrappers (thread-safe)
    # ------------------------------------------------------
    def d_move_forward(self, object_name):
        """Move the actor forward using the Blueprint function."""
        with self.lock:
            cmd = f"vbp {object_name} MoveForward"
            self.client.request(cmd)

    def d_set_max_speed(self, object_name, speed):
        """Set the max movement speed of the actor via Blueprint."""
        with self.lock:
            cmd = f"vbp {object_name} SetMaxSpeed {speed}"
            return self.client.request(cmd)

    def d_rotate(self, object_name, angle, direction="left"):
        """Rotate the actor by a given angle in a specified direction."""
        if direction == "right":
            clockwise = 1
        else:
            angle = -angle
            clockwise = -1
        with self.lock:
            cmd = f"vbp {object_name} Rotate_Angle {1} {angle} {clockwise}"
            self.client.request(cmd)

    def d_turn_around(self, object_name, angle, direction="left"):
        """Perform a turn-around action by rotating the actor."""
        if direction == "right":
            clockwise = 1
        else:
            angle = -angle
            clockwise = -1
        with self.lock:
            cmd = f"vbp {object_name} TurnAround {angle} {clockwise}"
            self.client.request(cmd)

    def d_step_forward(self, object_name, time, direction=None):
        """
        Step forward for a given duration using a Blueprint call.。
        """
        with self.lock:
            if direction is None:
                cmd = f"vbp {object_name} StepForward {time}"
            else:
                cmd = f"vbp {object_name} StepForward {direction} {time}"
            return self.client.request(cmd)

    def d_stop(self, object_name):
        """Stop the actor's movement."""
        with self.lock:
            cmd = f"vbp {object_name} StopDeliveryMan"
            self.client.request(cmd)

    def d_get_on_scooter(self, object_name):
        with self.lock:
            cmd = f"vbp {object_name} GetOnScooter"
            return self.client.request(cmd)
        
    def d_get_off_scooter(self, object_name):
        """Trigger the GetOffScooter Blueprint function on the scooter pawn."""
        with self.lock:
            cmd = f"vbp {object_name} GetOffScooter"
            return self.client.request(cmd)
        
    def d_enter_vehicle(self, humanoid_name, vehicle_name):
        """Enter vehicle.
        """
        with self.lock:
            cmd = f'vbp {humanoid_name} EnterVehicle {vehicle_name}'
            return self.client.request(cmd)
        
    def d_give(self, object_name):
        """Trigger the Give Blueprint function on the specified actor."""
        with self.lock:
            cmd = f"vbp {object_name} Give"
            return self.client.request(cmd)

    def s_set_state(self, object_name, throttle, brake, steering):
        """Set vehicle-like control parameters for the actor."""
        with self.lock:
            cmd = f"vbp {object_name} SetState {throttle} {brake} {steering}"
            self.client.request(cmd)

    def get_informations(self, manager_object_name):
        """Retrieve delivery-related information from the manager actor."""
        with self.lock:
            cmd = f"vbp {manager_object_name} GetDeliveryInformation"
            return self.client.request(cmd)

    def update_agents(self, manager_object_name):
        """Trigger an agent update routine on the manager actor."""
        with self.lock:
            cmd = f"vbp {manager_object_name} UpdateAgents"
            self.client.request(cmd)

    def making_u_turn(self, object_name):
        """Execute a U-turn action on the actor."""
        with self.lock:
            cmd = f"vbp {object_name} MakingUTurn"
            self.client.request(cmd)

    def d_pick_up(self, d_name, object_name):
        """Trigger a PickUp action between a delivery actor and an object."""
        with self.lock:
            cmd = f"vbp {d_name} PickUp {object_name}"
            self.client.request(cmd)

    def get_camera_observation(self, camera_id, viewmode="lit"):
        """
        Retrieve a camera observation using fast BMP mode.
        Returns a numpy array in BGR format.
        """
        return self.read_image(camera_id, viewmode, "fast")