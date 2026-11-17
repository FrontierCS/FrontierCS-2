- `development_mean_episode_reward`: Candidate policy's summed step reward until the first
  termination, mean over environments; mean over workloads.
- `development_mean_tracking_lin_vel`: Candidate policy's summed linear-velocity tracking reward
  term, mean over environments; mean over workloads.
- `development_mean_tracking_ang_vel`: Candidate policy's summed yaw-rate tracking reward term,
  mean over environments; mean over workloads.
- `development_mean_energy`: Candidate policy's per-alive-step energy penalty term (negative;
  closer to 0 is better); mean over workloads.
- `development_mean_action_rate`: Candidate policy's per-alive-step action-smoothness penalty
  term (negative; closer to 0 is better); mean over workloads.
