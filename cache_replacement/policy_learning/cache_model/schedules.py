#!/usr/bin/env python3

"""
A modified copy of https://github.com/openai/baselines/blob/master/baselines/common/schedules.py

This file was added to remove the big `baselines` dependency, as only a small part of it is used.
All the necessary pieces are copied here.
"""


class Schedule:
    def value(self, t):
        raise NotImplementedError()


class ConstantSchedule(Schedule):
    def __init__(self, value):
        self._v = value

    def value(self, t):
        return self._v


class LinearSchedule(Schedule):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
