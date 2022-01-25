import math
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from pprint import pprint
from utils import ArcIntervalTree

# %matplotlib inline
# import mpld3
# mpld3.enable_notebook()

ETA_1 = 1.0 # the ior "above surface"
ETA_2 = 1.5 # the ior "below surface"
SUR_RES = 100
ZOOM_SIZE = 5
CONTACT_ANGLE = 70

DISPLAY_TRACE = np.inf # index of trace to display, -1 to not display, np.inf for all traces
BOUNCE_COLOR = ['tab:blue','tab:orange', 'tab:green', 'tab:red', 'tab:pink']

class Ray:
    def __init__(self, origin, direction, wvl, complex_value, dist = np.Inf):
        """Create a ray with the given origin and direction."""
        self.origin = np.array(origin)
        self.direction = np.array(direction)
        self.wvl = wvl
        self.complex_value = complex_value
        self.dist = dist

# Rays going downwards
def generate_rays(ANGLE_I, spacing, NUM_RAYS, wavelength):
    rays = []
    phi = ANGLE_I / 180 * np.pi
    theta = phi - 1.5 * np.pi
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    ray_dir = np.array([math.cos(phi), math.sin(phi)])
    for num in range(NUM_RAYS):
        measure = (num + 0.5 - 0.5 * NUM_RAYS) * spacing
        unshifted_origin = np.array([measure * cos_theta, measure * sin_theta])
        shifted_origin = unshifted_origin - 10 * ray_dir
        new_ray = Ray(shifted_origin, ray_dir, wavelength, 1.0)
        print("Origin", shifted_origin)
        print("Direction", ray_dir)
        rays.append(new_ray)
    return rays

class LineSeg:
    def __init__(self, nodes, normals):
        self.nodes = nodes
        self.normals = normals

    def intersect(self, ray):
        """Computes the intersection between a ray and line segment, if it exists."""
        intersect_p, intersect_d, intersect_n = [], None, []
        total_segments = len(self.nodes) - 1
        for i in range(total_segments):
            p1 = self.nodes[i]
            p2 = self.nodes[i + 1]
            v1 = np.array([ray.origin - p1])
            v2 = np.array([p2 - p1])
            v3 = np.array([-ray.direction[1], ray.direction[0]])
            t1 = np.cross(v2, v1) / np.dot(v2, v3)
            t2 = np.dot(v1, v3) / np.dot(v2, v3)
            if t1 > 0.0 and t2 >= 0.0 and t2 < 1.0:
                point = ray.origin + t1 * ray.direction
                pt_normal = (1 - t2) * self.normals[i] + t2 * self.normals[i + 1]
                if intersect_d == None or t1 < intersect_d:
                    intersect_p, intersect_d, intersect_n = point, t1, pt_normal
#                     print("INTERSCTION_POINT", intersect_p)
#                     print("NORMAL", intersect_n)
        return intersect_p, intersect_d, intersect_n
    
class Trace:
    def __init__(self, ray_list, rt_seq):
        self.ray_list = ray_list
        self.rt_seq = rt_seq

def normalize(v):
    return v / np.linalg.norm(v)

def reflection_fresnel(cos_i, cos_t, eta_i, eta_t, s):
    rs = ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t))
    rp = ((eta_i * cos_t) - (eta_t * cos_i)) / ((eta_i * cos_t) + (eta_t * cos_i))
    return rs if s else rp

def transmission_fresnel(cos_i, cos_t, eta_i, eta_t, s):
    ts = 2 * eta_i * cos_i / ((eta_i * cos_i) + (eta_t * cos_t))
    tp = 2 * eta_i * cos_i / ((eta_i * cos_t) + (eta_t * cos_i))
    return ts if s else tp

def adj_intersect(intersect, direction):
    return intersect + 1e-10*direction

def perp_normal(p1, p2):
    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    return normalize([-dy, dx])

# Returns a list of traces, representing all the possible trajectories following an incident ray
# history is a Trace object
def paths(line_segments, current_ray, history, s):
    
    # See if the ray intersects with the surface
    intersection, dist, n = line_segments.intersect(current_ray)
    
    # No intersection found, the returned list contains a single Trace object as this path has ended
    if dist == None:
        final_list = history.ray_list + [current_ray]
        final_trace = Trace(final_list, history.rt_seq)
        return [final_trace]
    
    # If an intersection is found, we first determine which side this ray is coming from
    above_surface = True
    nl = n
    cos_i = -np.dot(n, current_ray.direction)
    if (cos_i < 0):
        above_surface = False
        cos_i = -cos_i
        nl = -nl
    
    # Compute the transmitted direction
    eta_i = ETA_1
    eta_t = ETA_2
    if not above_surface:
        eta_i = ETA_2
        eta_t = ETA_1
    ior_ratio = eta_t / eta_i
    sin_i = np.sqrt(1 - cos_i * cos_i)
    sin_t = sin_i / ior_ratio
    total_refl = False
    if (1 - sin_t * sin_t < 0):
        total_refl = True
    cos_t = np.sqrt(1 - sin_t * sin_t) # might be complex but that's okay
    
    # Prepare to update the trace history
    current_ray.dist = dist
    new_list = history.ray_list + [current_ray]
    
    # Consider reflection first (always considered)
    refl_dir = normalize(current_ray.direction + 2 * cos_i * nl)
    refl_fresnel = reflection_fresnel(cos_i, cos_t, eta_i, eta_t, s)
    refl_origin = adj_intersect(intersection, refl_dir)
    val1 = current_ray.complex_value * np.exp(-2j * np.pi * eta_i / current_ray.wvl * dist) * refl_fresnel
    ray1 = Ray(refl_origin, refl_dir, current_ray.wvl, val1)
    seq1 = history.rt_seq + "R"
    hist1 = Trace(new_list, seq1)
    r_traces = paths(line_segments, ray1, hist1, s)
    
    # Consider refraction if total internal reflection does not happen
    if total_refl:
        t_traces = []
    else:
        vec = normalize(current_ray.direction - np.dot(current_ray.direction, nl) * nl)
        tran_dir = normalize(-cos_t * nl + sin_t * vec)
        tran_fresnel = transmission_fresnel(cos_i, cos_t, eta_i, eta_t, s)
        tran_origin = adj_intersect(intersection, tran_dir)
        val2 = current_ray.complex_value * np.exp(-2j * np.pi * eta_i / current_ray.wvl * dist) * tran_fresnel
        ray2 = Ray(tran_origin, tran_dir, current_ray.wvl, val2)
        seq2 = history.rt_seq + "T"
        hist2 = Trace(new_list, seq2)
        t_traces = paths(line_segments, ray2, hist2, s)
        
    #Putting the results together
    return r_traces + t_traces

def plot_surface():
    def height(x):
        r = 2 / math.cos((90 - CONTACT_ANGLE) / 180 * np.pi)
        y = math.sqrt(r * r - 4) - math.sqrt(r * r - x * x)
        return y
    points = []
    normals = []
    i_range = range(-2 * SUR_RES, 2 * SUR_RES + 1)
    i_total = len(i_range)
    for i in i_range:
        j = i / SUR_RES
        points.append(np.array([j, height(j)]))
        curr_num = len(points)
        if curr_num == 2:
            normals.append(np.array(perp_normal(points[0], points[1])))
        if curr_num >= 3:
            normals.append(np.array(perp_normal(points[curr_num - 3], points[curr_num - 1])))
        if curr_num == i_total:
            normals.append(np.array(perp_normal(points[curr_num - 2], points[curr_num - 1])))
    lineseg = LineSeg(points, normals)
    for i in range(len(points)-1):
        x, y = [points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]]
        plt.plot(x, y, 'black')
    return lineseg

# Create a dictionary from a list of traces: the key is an RT-sequence, and the value is the last ray in the path
def trace_dictionary(traces):
    dict = {}
    for tr in traces:
        dict[tr.rt_seq] = tr.ray_list[-1]
    return dict

# Given two dictionary of traces, pair up the outgoing rays based on RT sequences
def find_pairs(dict1, dict2):
    ls1 = []
    ls2 = []
    for my_key in dict1:
        if my_key in dict2:
            ls1.append(dict1[my_key])
            ls2.append(dict2[my_key])
    return ls1, ls2

# Find the degree value representing the outgoing direction of a ray (use the conventional unit circle in math)
def find_degree(ray):
    ray_cos = ray.direction[0]
    ray_sin = ray.direction[1]
    ray_acos = np.arccos(ray_cos)
    ray_asin = np.arcsin(ray_sin)
    if (ray_acos <= np.pi / 2):
        if (ray_asin >= 0):
            return ray_asin / np.pi * 180
        else:
            return 360 + ray_asin / np.pi * 180
    else:
        if (ray_asin >= 0):
            return ray_acos / np.pi * 180
        else:
            return 360 - ray_acos / np.pi * 180

def calculate_interference(query, interval_set, spacing, interpolate_scheme):
    if len(interval_set) == 0:
        return 0
    #print("QUERY", query)
    # Start to compute the sum of a bunch of complex numbers
    field_val = 0
    for ang1, ang2, data in interval_set:
        #print("ANG1", ang1)
        #print("ANG2", ang2)
        ray1 = data[0]
        ray2 = data[1]
        diff = ang2 - ang1
        if diff == 0:
            raise Exception("This shouldn't have happened")
        else:
            w1 = (ang2 - query) / diff
            w2 = (query - ang1) / diff
            scale = spacing / diff
            my_eta = ETA_1 if ang1 <= 180 else ETA_2
            val1 = ray1.complex_value * np.exp(-2j * np.pi * my_eta / ray1.wvl * ray1.dist)
            #print("VALUE 1", val1)
            val2 = ray2.complex_value * np.exp(-2j * np.pi * my_eta / ray2.wvl * ray2.dist)
            #print("VALUE 2", val2)
            if interpolate_scheme == 1:
                val = w1 * val1 + w2 * val2
                #print("VALUE", val)
            else:
                new_abs = w1 * abs(val1) + w2 * abs(val2)
                new_phase = w1 * cmath.phase(val1) + w2 * cmath.phase(val2)
                val = new_abs * np.exp(1j * new_phase)
                #print("VALUE", val)
            field_val = field_val + val * np.sqrt(scale)
    # return the computed complex number
    return field_val

def split_line(ray):
    rs = []
    rs.append(ray.origin)

    dist = ray.dist if ray.dist != np.inf else 100
    rs.append(ray.origin+dist*ray.direction)

    return [x for x, y in rs], [y for x, y in rs]

# def get_markers(ray):
#     rs = []
#     dist = ray.dist if ray.dist != np.inf else 100

#     phase_ratio = ray.phase_offset/(2*math.pi)
#     if (1-phase_ratio)*ray.wavelength < dist:
#         rs.append(ray.origin+(1-phase_ratio)*ray.wavelength*ray.direction)
    
#     i = 0
#     while i < dist-(1-phase_ratio)*ray.wavelength:
#         rs.append(rs[0]+i*ray.direction)
#         i = i+ray.wavelength
        
#     return [x for x, y in rs], [y for x, y in rs]

def draw_rays(ray, color):
    xs, ys = split_line(ray)
#     xm, ym = get_markers(ray)
    plt.plot(xs, ys, alpha=1, color=color)

    p = ray.direction
#     plt.plot(xm, ym, linestyle = 'None', alpha=RAY_OPACITY, marker=[(-p[1], p[0]), (p[1], -p[0])], markersize=5+5*np.abs(ray.complex_value), mec=color)


def plot_traces(ax, ray, traces):
    plt.xlim([-ZOOM_SIZE, ZOOM_SIZE])
    plt.ylim([-ZOOM_SIZE, ZOOM_SIZE])
    plot_surface()
    for trace in traces:
        rays = trace.ray_list
        trace_len = len(rays)
        num_bounce = len(rays)-1

        draw_rays(ray, "grey")

        if trace_len > 0:
#             print(rays)
            for i in range(1, trace_len):
                r = rays[i]
                draw_rays(r, BOUNCE_COLOR[(num_bounce-1)%5])

fig, ax = plt.subplots(figsize=(6, 6))
plot1 = plt.figure(1)
plt.xlim([-ZOOM_SIZE, ZOOM_SIZE])
plt.ylim([-ZOOM_SIZE, ZOOM_SIZE])
ANGLE_I = 270
NUM_RAYS = 20
spacing = 3.99 / (NUM_RAYS - 1)
wavelength = 0.5
rays = generate_rays(ANGLE_I, spacing, NUM_RAYS, wavelength)
lineseg = plot_surface()

tree = ArcIntervalTree()
empty_hist = Trace([], "")
prev_traces = paths(lineseg, rays[0], empty_hist, True)
if DISPLAY_TRACE == np.inf or DISPLAY_TRACE == 0:
    plot_traces(ax, rays[0], prev_traces)
prev_dict = trace_dictionary(prev_traces)
for count in range(1, NUM_RAYS):
    #print("Count", count)
    curr_traces = paths(lineseg, rays[count], empty_hist, True)
    if DISPLAY_TRACE == np.inf or DISPLAY_TRACE == count:
        plot_traces(ax, rays[count], curr_traces)
    curr_dict = trace_dictionary(curr_traces)
    #print("Reflected ray origin", curr_dict["R"].origin)
    #print("Reflected ray direction", curr_dict["R"].direction)
    #print("Transmitted ray origin", curr_dict["T"].origin)
    #print("Transmitted ray direction", curr_dict["T"].direction)
    rays1, rays2 = find_pairs(prev_dict, curr_dict)
    num_pairs = len(rays1)
    for pair_count in range(num_pairs):
        #print("Pair", pair_count + 1)
        ang1 = find_degree(rays1[pair_count])
        ang2 = find_degree(rays2[pair_count])
        #print("Angle 1", ang1)
        #print("Angle 2", ang2)
        rays1[pair_count].dist = -np.dot(rays1[pair_count].origin, rays1[pair_count].direction)
        rays2[pair_count].dist = -np.dot(rays2[pair_count].origin, rays2[pair_count].direction)
        tree.add_interval(ang1, ang2, (rays1[pair_count], rays2[pair_count]))
    prev_dict = curr_dict
query_pts = []
intensities = []
intensity_refl = 0
intensity_tran = 0
ARC_RES = 50
for i in range(360 * ARC_RES):
    query = i / ARC_RES
    query_pts.append(query)
    field_val = calculate_interference(query, tree.get_intervals(query), spacing, 1)
    intensity = np.abs(field_val) ** 2
    intensities.append(intensity)
#     if (intensity > 0):
#         print("Intensity", intensity)
    if query <= 180:
        intensity_refl = intensity_refl + intensity / ARC_RES
    else:
        intensity_tran = intensity_tran + intensity / ARC_RES
    
intensity_inci = spacing * (NUM_RAYS - 1) * 1.0 * ETA_1
intensity_refl = intensity_refl * ETA_1
intensity_tran = intensity_tran * ETA_2
intensity_total = intensity_refl + intensity_tran
error = abs(intensity_total - intensity_inci) / intensity_inci
print("INCIDENT ENERGY", intensity_inci)
print("REFLECTED ENERGY", intensity_refl)
print("TRANSMITTED ENERGY", intensity_tran)
print("RESULTING ENERGY", intensity_refl + intensity_tran)
print("ERROR IN ENERGY", error)

plt.show()