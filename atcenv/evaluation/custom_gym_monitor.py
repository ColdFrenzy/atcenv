""" Custom gym monitor is needed because standard gym monitor doesn't suypport
multi-agent i.e. dict.
"""
import gym


class CustomGymMonitor(gym.wrappers.Monitor):
    def __init__(self, **kwargs):
        super(CustomGymMonitor, self).__init__(**kwargs)

    def _after_step(self, observation, reward, done, info):
        if not self.enabled:
            return done

        # if done and self.env_semantics_autoreset:
        #     # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
        #     # self.reset_video_recorder()
        #     self.episode_id += 1
        #     self._flush()

        # Don't record stats
        # self.stats_recorder.after_step(observation, reward, done, info)
        # Record video
        self.video_recorder.capture_frame()

        return done
