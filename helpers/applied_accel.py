"""Applied acceleration of the simulated car.

DPC_collect_data.py logs RaceCar.accel as the acceleration input of the DPC data. The gym version the datasets in
data/dpc_experiments were recorded with stored there the acceleration the vehicle actually receives: in 'acc' drive mode
the command clipped to +-a_max, then limited by accl_constraints (power limit above v_switch) at the speed before the
step. The multibody gym fork (f1tenth_gym, branch multibody, e55925e) never sets it (stays 0). install() restores it, so
recordings reproduce the committed datasets exactly and DPC.py can feed its input window the same quantity.
"""
import numpy as np
import f110_gym.envs.base_classes as bc


def install():
    if getattr(bc.RaceCar.update_pose, "_stores_applied_accel", False):
        return
    update_pose_gym = bc.RaceCar.update_pose

    def update_pose(self, raw_steer, drive):
        v0 = float(self.state[3])
        result = update_pose_gym(self, raw_steer, drive)
        if self.drive_control_mode == 'acc':
            p = self.params
            self.accel = float(bc.accl_constraints(v0, float(np.clip(drive, -p['a_max'], p['a_max'])),
                                                   p['v_switch'], p['a_max'], p['v_min'], p['v_max']))
        return result

    update_pose._stores_applied_accel = True
    bc.RaceCar.update_pose = update_pose
