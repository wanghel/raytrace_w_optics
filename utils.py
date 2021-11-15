

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
from intervaltree import Interval, IntervalTree


class ArcIntervalTree(IntervalTree):
    def add_interval(self, ang1, ang2, data):
        ang1 = ang1%360
        ang2 = ang2%360

        # flip data so (small, big)
        if ang2 - ang1 <= 0:
            data = (data[1], data[0])
            temp = ang1
            ang1 = ang2
            ang2 = temp

        # if np.abs(ang2 - ang1) >= 180:
        #     if ang1 >= 359:
        #         self[ang2-360:0] = data
        #         print("ang1:", ang2-360)
        #         print("ang2:", 0)
        #     else:
        #         self[ang2-360:ang1+1] = data
        #         print("ang1:", ang2-360)
        #         print("ang2:", (ang1+1))
        # else: 
        #     if ang2 >= 359:
        #         print("ang1:", ang1-360)
        #         print("ang2:", 0)
        #         self[ang1-360:0] = data
        #     else:
        #         print("ang1:", ang1)
        #         print("ang2:", (ang2+1))
        #         self[ang1:ang2+1] = data
        if np.abs(ang2 - ang1) >= 180:
            if ang1 >= 359:
                self[ang2-360:0] = data
                print("ang1:", ang2-360)
                print("ang2:", 0)
            else:
                self[ang2-360:ang1] = data
                print("ang1:", ang2-360)
                print("ang2:", (ang1))
        else: 
            if ang2 >= 359:
                print("ang1:", ang1-360)
                print("ang2:", 0)
                self[ang1-360:0] = data
            else:
                print("ang1:", ang1)
                print("ang2:", (ang2))
                self[ang1:ang2] = data

    def get_intervals(self, point):
        point = point%360
        return set.union(self[point], self[point-360])

