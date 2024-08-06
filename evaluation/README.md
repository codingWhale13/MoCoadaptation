# Evaluation

The folder `old_code` contains evaluation scripts from the parent repository [psyberprimate/MoCoadaptation](https://github.com/psyberprimate/MoCoadaptation).

The files directly in the `evaluation` folder are a re-implementation and go together with the updated config format of [this repo](https://github.com/codingWhale13/MoCoadaptation).

## File Overviews

### Plotting

* `test_latest.py`:             Runs tests on the last model checkpoint for each run => required for plotting the approximate Pareto front
* `test_per_iteration.py`:      Runs tests for each design iteration of each run => required for plotting the reward over time
* `plot_approx_pf.py`:          Plots the approximate Pareto front (only 2D scenario currently supported)
* `plot_reward_over_time.py`:   Plots the reward per design cycle (skipping initial and random designs by default for clarity)

### Videos

* `video_from_checkpoint.py`:   Generates a video of one episode, using a determinstic policy loaded from a specific model
* `videos_from_checkpoint.py`:  Create multiple videos by calling `video_from_checkpoint` repeatedly (reason for doing it like this: just instantiating a new env and video player results in all videos but the first being mesed up => camera doesn't track agent correctly such that it's not visible most of the time)
* `videos_grid.py`:             Takes multiple videos (created by `video(s)_from_checkpoint.py`) as input and arranges them in a grid
