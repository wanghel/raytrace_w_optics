import math
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from pprint import pprint
# from intervaltree import Interval, IntervalTree
from utils import ArcIntervalTree
from binarytree import Node


ANGLE_I = math.radians(0.0)
IOR = 1.5
NUM_RAYS = 10
RAY_OPACITY = 1
ARC_RES = 1000
SUR_RES = 77
ZOOM_SIZE = 5

BOUNCE_COLOR = ['tab:blue','tab:orange', 'tab:green', 'tab:red', 'tab:pink']

class Ray:
    def __init__(self, origin, direction, start=0., end=np.inf, phase_offset=0, dist=np.inf, wavelength=0.5, amp=1, phasor=None):
        """Create a ray with the given origin and direction."""
        self.origin = np.array(origin)
        self.direction = np.array(direction)
        self.start = start
        self.end = end
        self.t = dist

        self.phase_offset = phase_offset
        self.wavelength = wavelength
        self.phasor = 1*np.exp(-1j*(2*np.pi/wavelength*0)) if phasor == None else phasor

    def get_end_phase_offest(self):
        print("OFFSET:", cmath.phase(self.phasor))
        return cmath.phase(self.phasor)

    def get_amp(self):
        return np.abs(self.phasor)
        
class LineSeg:
    def __init__(self, num, nodes, normals, eta):
        self.num = num
        self.nodes = nodes
        self.normals = normals
        self.eta = eta

    def intersect(self, ray):
        """Computes the intersection between a ray and line segment, if it exists."""
        t, eps, d = 1e20, 1e-10, 1e20
        
        intersect_p, intersect_t, seg_num = [], None, None

        for i in range(len(self.nodes)-1):
            p1 = self.nodes[i]
            p2 = self.nodes[i+1]

            v1 = np.array([ray.origin - p1])
            v2 = np.array([p2 - p1])
            v3 = np.array([-ray.direction[1], ray.direction[0]])

            # if np.dot(v2, v3) < 1e-10 and np.dot(v2, v3) > -1e-10:
                # return [], None, None

            t1 = np.cross(v2, v1) / np.dot(v2, v3)
            t2 = np.dot(v1, v3) / np.dot(v2, v3)

            if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
                point = ray.origin + t1*ray.direction

                if not np.array_equal(point, ray.origin):
                    # return point, t1, i
                    if intersect_t == None or t1 < intersect_t:
                        intersect_p, intersect_t, seg_num = point, t1, i
                        print("INTERSCTION_POINT", intersect_p)
                        print("NORMAL", self.normals[i])
                        print("SEG NUM", i)
        
        # return [], None, None
        return intersect_p, intersect_t, seg_num

class TraceInfo:
    def __init__(self, ray, t=0, weight=1):
        self.rays = Node(ray)
        self.tot_dist = t
        self.weight = weight
        self.num = 1

def normalize(v):
    return v / np.linalg.norm(v)

def FrDielecric(costhetai, costhetat, etai_parl, etat_parl, etai_perp, etat_perp, entering, perp):
    if not entering:
        temp = etai_parl
        etai_parl = etat_parl
        etat_parl = temp

        temp = etai_perp
        etai_perp = etat_perp
        etat_perp = temp

    rs = ((etai_perp*costhetai) - (etat_perp*costhetat)) / ((etai_perp*costhetai) + (etat_perp*costhetat)) # Rs
    # ts = (2*etai_perp*costhetai)/(etai_perp*costhetai + etat_perp*costhetat)

    rp = ((etat_parl*costhetai) - (etai_parl*costhetat)) / ((etat_parl*costhetai) + (etai_parl*costhetat)) # Rp
    # tp = (2*etai_parl*costhetai)/(etat_parl*costhetai + etai_parl*costhetat)
    
    # return rs*rs if perp else rp*rp
    # ONLY REFLECTTION
    return 1
    # TAKE AVG OF POLARIZATIONS
    # return (rs*rs + rp*rp) / 2

def reflection_fresnel(costhetai, costhetat, etai_parl, etat_parl, etai_perp, etat_perp, entering, perp):
    if not entering:
        temp = etai_parl
        etai_parl = etat_parl
        etat_parl = temp

        temp = etai_perp
        etai_perp = etat_perp
        etat_perp = temp

    rs = ((etai_perp*costhetai) - (etat_perp*costhetat)) / ((etai_perp*costhetai) + (etat_perp*costhetat)) # Rs
    # ts = (2*etai_perp*costhetai)/(etai_perp*costhetai + etat_perp*costhetat)

    rp = ((etat_parl*costhetai) - (etai_parl*costhetat)) / ((etat_parl*costhetai) + (etai_parl*costhetat)) # Rp
    # tp = (2*etai_parl*costhetai)/(etat_parl*costhetai + etai_parl*costhetat)
    
    # return rs if perp else rp
    return 1

def transmission_fresnel(costhetai, costhetat, etai_parl, etat_parl, etai_perp, etat_perp, entering, perp):
    if not entering:
        temp = etai_parl
        etai_parl = etat_parl
        etat_parl = temp

        temp = etai_perp
        etai_perp = etat_perp
        etat_perp = temp

    rs = 2*(etai_perp*costhetai) / ((etai_perp*costhetai) + (etat_perp*costhetat)) # Rs
    # ts = (2*etai_perp*costhetai)/(etai_perp*costhetai + etat_perp*costhetat)

    rp = 2*(etai_parl*costhetai) / ((etat_parl*costhetai) + (etai_parl*costhetat)) # Rp
    # tp = (2*etai_parl*costhetai)/(etat_parl*costhetai + etai_parl*costhetat)
    
    # return rs if perp else rp
    return 1

def adj_intersect(intersect, direction):
    return intersect+1e-10*direction

def radiance(trace, ray, ls, depth, RRprob, weight, total_dist):
    intersection, dist, num_ls = ls.intersect(ray)
    if dist != None:
        total_dist = total_dist + dist
        # trace.value.
    # print("WEIGHT", weight)
    # print("ray origin", ray.origin)
    # print("ray direction", ray.direction)
    # print("intersection", intersection)
    # print("num_ls", num_ls)
    if (intersection == [] or depth > 0):
        return Node(TraceInfo(ray, weight=weight))

    ratio = 0
    n = None
    if num_ls == ls.num-1:
        n = ls.normals[ls.num - 1]*(1 - ratio) + ls.normals[0]*ratio
    else:
        n = ls.normals[num_ls]*(1 - ratio) + ls.normals[num_ls + 1]*ratio
    
    nl = n if np.dot(n, ray.direction) < 0 else -n
    into = np.dot(n, nl) > 0

    nc = 1.0
    nt = ls.eta
    nnt = nc/nt if into else nt/nc
    ddn = np.dot(ray.direction, nl)
    cos2t = 1 - nnt*nnt*(1 - ddn*ddn)

    fresnel = 1
    if cos2t > 0:
        tdir = normalize((ray.direction*nnt - n*((1 if into else -1)*(ddn*nnt + math.sqrt(cos2t)))))
        costhetai = abs(np.dot(nl, ray.direction))
        costhetat = abs(np.dot(nl, tdir))
        fresnel = FrDielecric(costhetai, costhetat, 1.0, ls.eta, 1.0, ls.eta, into, True)

    # russian roulette
    depth = depth + 1
    if depth > 5:
        if random.random() > RRprob:
            return Node(TraceInfo(ray, weight=weight))
        else:
            weight = weight / RRprob
    
    # reflection
    fresnel = reflection_fresnel(costhetai, costhetat, 1.0, ls.eta, 1.0, ls.eta, into, True)
    
    new_dir = normalize(ray.direction - n*2*np.dot(n,ray.direction))
    intersection = adj_intersect(intersection, new_dir)
    ray.end = intersection
    ray.t = dist
    trace.value.dist = dist
    trace.value.ray = ray
    trace.value.weight = weight

    new_phasor = fresnel*ray.get_amp()*np.exp(-1j*(2*np.pi/ray.wavelength*total_dist))
    r = Ray(intersection, new_dir, phase_offset=ray.get_end_phase_offest(), phasor=new_phasor)
    new_trace = TraceInfo(ray)
    new_trace = radiance(new_trace, r, ls, depth, RRprob, weight, total_dist)
    trace.left = new_trace

    # refraction
    if cos2t < 0:
        cos2t = -cos2t
    tdir = normalize((ray.direction*nnt - n*((1 if into else -1)*(ddn*nnt + math.sqrt(cos2t)))))
    costhetai = abs(np.dot(nl, ray.direction))
    costhetat = abs(np.dot(nl, tdir))
    fresnel = transmission_fresnel(costhetai, costhetat, 1.0, ls.eta, 1.0, ls.eta, into, True)

    tdir = normalize((ray.direction*nnt - n*((1 if into else -1)*(ddn*nnt + math.sqrt(cos2t)))))
    intersection = adj_intersect(intersection, tdir)
    ray.end = intersection
    ray.t = dist
    trace.value.dist = dist
    trace.value.ray = ray
    trace.value.weight = weight

    new_phasor = fresnel*ray.get_amp()*np.exp(-1j*(2*np.pi/ray.wavelength*total_dist))
    r = Ray(intersection, tdir, phase_offset=ray.get_end_phase_offest(), phasor=new_phasor)
    new_trace = TraceInfo(ray)
    new_trace = radiance(new_trace, r, ls, depth, RRprob, weight, total_dist)
    trace.right = new_trace

    return trace

def perp_normal(p1, p2):
    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    
    return normalize([-dy, dx])

def collect_bin_ang(d):
    ang = math.acos(np.dot(np.array(normalize([1, 0])), d))
    # print("ANG", math.degrees(ang))
    ang = (math.pi/2-ang)%(2*math.pi)
    # if d[1] < 0:
    #     ang = -ang+2*math.pi
    # ang = (ang-math.pi)%(2*math.pi)
    return math.degrees(ang)

# def generate_ray(ang, num):
#     if NUM_RAYS <= 1:
#         raise Exception("NUM_RAYS must be greater than 1")
#     # ox = num/NUM_RAYS-3
#     oy = 4*num/(NUM_RAYS-1)-2
#     # print("cos", math.tan(ANGLE_I))
#     ox = -5+num/NUM_RAYS*math.tan(ANGLE_I)
#     print("RAY ORIGIN", ox, oy)
#     origin = np.array([ox, oy])
#     ray_dir = normalize(np.array([math.cos(ang), math.sin(ang)]))

#     return Ray(origin, ray_dir)

# Rays going downwards
def generate_ray(ang, num):
    # ox = num/NUM_RAYS-3
    ox = num/NUM_RAYS
    # if num == 0:
    #     ox = 0
    # elif num == 1:
    #     ox = 0.3333333333333333
    # print(ox)
    # print("cos", math.tan(ANGLE_I))
    oy = 1+num/NUM_RAYS*math.tan(ANGLE_I)
    print("RAY ORIGIN", ox, oy)
    origin = np.array([ox, oy])
    ray_dir = normalize(np.array([math.sin(ang), -math.cos(ang)]))

    return Ray(origin, ray_dir)

# Random rays [0.0, 1.0]
# def generate_ray(ang):
#     ox = (random.random())
#     oy = 1
#     origin = np.array([ox, oy])
#     ray_dir = normalize(np.array([9.99999809e-01-ang, -6.18144403e-04-ang]))
#     return Ray(origin, ray_dir)

# Gaussian ray
# def generate_ray(ang, num):
#     wavelength = 650*1e-9
#     waist = 170*1e-6
#     std = waist/2

#     cury = np.random.normal(0, std)
#     origin = np.array([0.0, cury])
#     divergence = wavelength/(math.sqrt(2)*math.pi*waist)

#     beam_angle = np.random.normal(0, divergence/math.sqrt(2))
#     direction = normalize(np.array([math.cos(beam_angle)-ang, math.sin(beam_angle)-ang]))

#     ray =  Ray(origin, direction) 

#     return ray

def split_line(ray):
    rs = []
    rs.append(ray.origin)

    t = ray.t if ray.t != np.inf else 100
    rs.append(ray.origin+t*ray.direction)

    return [x for x, y in rs], [y for x, y in rs]

def get_markers(ray):
    rs = []
    t = ray.t if ray.t != np.inf else 100

    phase_ratio = ray.phase_offset/(2*math.pi)
    if (1-phase_ratio)*ray.wavelength < t:
        rs.append(ray.origin+(1-phase_ratio)*ray.wavelength*ray.direction)
    
    i = 0
    while i < t-(1-phase_ratio)*ray.wavelength:
        rs.append(rs[0]+i*ray.direction)
        i = i+ray.wavelength
        
    return [x for x, y in rs], [y for x, y in rs]

def draw_rays(ray, color):
    xs, ys = split_line(ray)
    xm, ym = get_markers(ray)
    # plt.plot(xs, ys, alpha=0.3, color=color)

    p = ray.direction
    plt.plot(xm, ym, linestyle = 'None', alpha=RAY_OPACITY, marker=[(-p[1], p[0]), (p[1], -p[0])], markersize=5+5*ray.get_amp(), mec=color)


def plot_trace(ax, ray, return_trace, collect_circ):
    rrays = return_trace.rays
    trace_len = len(rrays)
    num_bounce = return_trace.num - 1

    draw_rays(rrays[num_bounce], "grey")
    # print("FIRST origin", ray.origin)
    # print("FIRST direction", ray.direction)

    if trace_len > 0:
        # print("origin", rrays[trace_len-1].origin)
        # print("direction", rrays[trace_len-1].direction)

        for i in range(1, num_bounce):
            r = rrays[trace_len-1-i]
            draw_rays(r, BOUNCE_COLOR[(num_bounce-1)%5])
            
    draw_rays(rrays[0], BOUNCE_COLOR[(num_bounce-1)%5])

    ang = int(collect_bin_ang(rrays[0].direction))
    if ang in collect_circ:
        collect_circ[ang] = collect_circ[ang] + return_trace.weight        
    else:
        collect_circ[ang] = return_trace.weight

    return collect_circ, collect_bin_ang(rrays[0].direction)

def plot_surface():
    def height(x):
        # y = -2 
        # y = (x/5)**2*2-2
        # y = -math.sqrt(4-x**2)
        y = math.sin(x*10)/50-1
        # y = math.sin(x*2)-1
        return y + random.random()*1e-5

    points = []
    normals = []
    for i in range(-10*SUR_RES, 10*SUR_RES):
        j = i/SUR_RES
        points.append(np.array([j, height(j)]))

        if len(points) > 1:
            normals.append(np.array(perp_normal(points[len(points)-2], points[len(points)-1])))

    # make circle
    if False:
        for i in range(-10*SUR_RES, 10*SUR_RES):
            j = i/SUR_RES
            points.append(np.array([-j, -height(j)]))

            if len(points) > 1:
                normals.append(np.array(perp_normal(points[len(points)-2], points[len(points)-1])))

        points.append(points[0])
        normals.append(np.array(perp_normal(points[len(points)-2], points[len(points)-1])))

    lineseg = LineSeg(len(normals), points, normals, IOR)

    for i in range(len(points)-1):
        x, y = [points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]]
        plt.plot(x, y, 'black')
        
    return lineseg

def ang_diff(ang1, ang2):
    diff = (ang2-ang1+180)%360-180
    return diff+360 if diff < -180 else diff

def calculate_interference(ang, interval_set):
    intf = 0 
    intrpol_amp = 0
    if len(interval_set) == 0:
        return intf, intrpol_amp

    for ang1, ang2, data in interval_set:
        trace1 = data[0]
        trace2 = data[1]

        rray1 = trace1.rays[0]
        rray2 = trace2.rays[0]

        iray1 = trace1.rays[-1]
        iray2 = trace2.rays[-1]

        # diff = ang2-1-ang1
        diff = (ang2-ang1)
        # print("DIFF", diff)

        if diff == 0:
            raise Exception("rare case of ang being same happened, should've crashed already")
            # intf = intf + (rray1.phasor+rray2.phasor)/2
        else:
            # w1 = (ang2 - ang)/(ang2 - ang1)
            # w2 = (ang - ang1)/(ang2 - ang1)
            w1 = ang_diff(ang, ang2)/ang_diff(ang1, ang2)
            w2 = ang_diff(ang1, ang)/ang_diff(ang1, ang2)
            if np.abs(w1+w2-1) > 1e-5:
                raise Exception("Weights w1, w2 don't add up to 1.0")
                
            # print("ORG DIFF", np.linalg.norm(iray2.origin-iray1.origin))
            arc_w = np.linalg.norm(iray2.origin-iray1.origin)/diff
            # print("arc_w", arc_w)
            # print("rray1", np.abs(rray1.phasor)**2)
            # print("rray2", np.abs(rray2.phasor)**2)
            
            # print("AMPLITUDE", w1*np.abs(rray1.phasor) + w2*np.abs(rray2.phasor))
            intf = intf + (w1*rray1.phasor + w2*rray2.phasor)*math.sqrt(arc_w)
            # intrpol_amp = intrpol_amp + (w1*np.abs(rray1.phasor) + w2*np.abs(rray2.phasor))*math.sqrt(arc_w)
            intrpol_amp = intrpol_amp + (np.abs(w1*rray1.phasor + w2*rray2.phasor)*math.sqrt(arc_w))**2
            # print("phasor 1", np.abs(rray1.phasor), rray1.phasor)
            # print("phasor 2", np.abs(rray2.phasor), rray2.phasor)
            # print("AMP", np.abs((w1*rray1.phasor + w2*rray2.phasor))**2)
            # print("AMP1", np.abs(rray1.phasor)**2)
            # print("AMP2", np.abs(rray2.phasor)**2)

    intf = intf/math.sqrt(len(interval_set))

    return intf, intrpol_amp
    
def makeplot():
    fig, ax = plt.subplots(figsize=(6,6))
    plot1 = plt.figure(1)
    plt.xlim([-ZOOM_SIZE, ZOOM_SIZE])
    plt.ylim([-ZOOM_SIZE, ZOOM_SIZE])

    collect_circ = dict([])

    rays = []
    
    for i in range(NUM_RAYS):
        # ray_pairs.append((generate_ray(ANGLE_I, i), generate_ray(ANGLE_I, i+1)))
        rays.append(generate_ray(ANGLE_I, i))


    lineseg = plot_surface()

    tree = ArcIntervalTree()

    num_rays_hit = 0
    tot_weight = 0

    prev_trace = None
    prev_ang = None

    for ray in rays:
        # print("ray origin", ray.origin)
        # print("ray direction", ray.direction)
        root = TraceInfo(ray)
        trace = Node(root)
        return_trace = radiance(root, ray, lineseg, 0, 0.95, 1, 0)
        
        if return_trace.height > 0:
            print("---return trace is greater than 1---")
            # print("RETURN", return_trace.tot_dist)
            return_ray = return_trace.rays[0]
            num_rays_hit = num_rays_hit + 1
            # print("return ray origin", return_ray.origin)
            # print("return ray direction", return_ray.direction)

            collect_circ, ang = plot_trace(ax, ray, return_trace, collect_circ)
            tot_weight = tot_weight + return_trace.weight
            
            if prev_trace != None:
                tree.add_interval(prev_ang, ang, (prev_trace, return_trace))

            prev_trace = return_trace
            print("ANG", ang)
            prev_ang = ang

    # print("tree", len(tree))
    print("INCIDENT ENERGY", np.linalg.norm(rays[0].origin-rays[-1].origin))
    # tree intervals are [beg, end)
    # a = tree.get_intervals(45)
    # for i in a:
    #     print("INTERVAL", i)

    xs = []
    ys = []
    sum_intensity = 0
    sum_intensity_from_amp = 0
    print(tree)
    for i in range(360*ARC_RES):
        arc = i/ARC_RES
        xs.append(arc)
        x, x1 = calculate_interference(arc, tree.get_intervals(arc))
        ys.append(np.abs(x)**2)
        if x!=0: 
            # print("angle intensity:", arc, (np.abs(x)**2)/ARC_RES)
            sum_intensity = sum_intensity + (np.abs(x)**2)/ARC_RES
            sum_intensity_from_amp = sum_intensity_from_amp + x1/ARC_RES
    print("ENERGY", sum_intensity)
    print("ENERGY INTRPOL AMP", sum_intensity_from_amp)
    plot2 = plt.figure(2)
    plt.plot(xs, ys)

    distr_circ = dict([])
    for k, v in collect_circ.items():
        distr_circ[k] = v/tot_weight
    pprint(distr_circ)


    plot3 = plt.figure(3)
    xs = list(range(360))
    ys = []
    for i in xs:
        if i in distr_circ:
            ys.append(distr_circ[i])
        else:
            ys.append(0)
    plt.hist(xs, 360, weights=ys)

    plt.show()

makeplot()