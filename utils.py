import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
from intervaltree import Interval, IntervalTree

class ArcIntervalTree(IntervalTree):
    def add_interval(self, ang1, ang2, data):
        # Weird case: one ray goes above the surface and the other goes below--this shouldn't happen
        if (ang1 - 180) * (ang2 - 180) < 0:
            return
        # Flip the angles if needed
        if ang2 < ang1:
            data = (data[1], data[0])
            temp = ang1
            ang1 = ang2
            ang2 = temp
        self[ang1:ang2] = data

    def get_intervals(self, point):
        # It can be assumed that 0 <= point < 360
        return self[point]