import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add the parent directory to sys.path
import math
from utils import in_circumcircle, angle


def get_neighbors(point, edges):
    neighbors = set()
    for edge in edges:
        if point in edge:
            neighbors.add(edge[0] if edge[1] == point else edge[1])
    return neighbors


def delaunay(points):
    points = sorted(set(points))  # Remove duplicates and sort by x, then y

    def build(points):
        n = len(points)
        if n == 2:
            return [(points[0], points[1])]
        if n == 3:
            return [(points[0], points[1]), (points[1], points[2]), (points[2], points[0])]

        mid = n // 2
        left_edges = build(points[:mid])
        right_edges = build(points[mid:])
        return merge(left_edges, right_edges, points[:mid], points[mid:])

    def merge(left_edges, right_edges, left_points, right_points):
        edges = left_edges + right_edges

        # Find the base LR-edge
        l_base = max(left_points, key=lambda p: p[0])  # rightmost point in left_points
        r_base = min(right_points, key=lambda p: p[0])  # leftmost point in right_points

        while True:
            updated = False

            # Check if there is a better l neighbor
            l_neighbors = get_neighbors(l_base, left_edges)
            for l_next in l_neighbors:
                theta = angle(l_next, l_base, r_base)
                if theta < math.pi:
                    l_base = l_next
                    updated = True
                    break  # take one step at a time

            # Check if there is a better r neighbor
            r_neighbors = get_neighbors(r_base, right_edges)
            for r_next in r_neighbors:
                theta = angle(r_next, l_base, r_base)
                if theta < math.pi:
                    r_base = r_next
                    updated = True
                    break

            if not updated:
                break

        base = (l_base, r_base)
        edges.append(base)

        while True:
            left_candidates = get_neighbors(base[0], left_edges)
            left_candidates = sorted(left_candidates, key=lambda c: angle(base[1], base[0], c))  # Sort by angle
            left_candidates = [c for c in left_candidates if angle(base[1], base[0], c) < math.pi]  # Remove the left_edges candidates which angle is >= than pi

            left_candidate = None
            while left_candidates:
                left_candidate = left_candidates.pop(0)
                if not left_candidates:
                    break
                left_candidate_next = left_candidates[0]
                if not in_circumcircle(left_candidate_next, left_candidate, base[0], base[1]):
                    break
                else:  # Remove the LL-edge
                    [edges.remove(edge) for edge in [(base[0], left_candidate), (left_candidate, base[0])] if edge in edges]

            right_candidates = get_neighbors(base[1], right_edges)
            right_candidates = sorted(right_candidates, key=lambda c: angle(c, base[1], base[0]))  # Sort by angle
            right_candidates = [c for c in right_candidates if angle(c, base[1], base[0]) < math.pi]  # Remove the right_edges candidates which angle is >= than pi

            right_candidate = None
            while right_candidates:
                right_candidate = right_candidates.pop(0)
                if not right_candidates:
                    break
                right_candidate_next = right_candidates[0]
                if not in_circumcircle(right_candidate_next, right_candidate, base[0], base[1]):
                    break
                else:  # Remove the RR-edge
                    [edges.remove(edge) for edge in [(base[1], right_candidate), (right_candidate, base[1])] if edge in edges]

            if left_candidate == None and right_candidate == None:
                break

            if left_candidate == None:
                base = (base[0], right_candidate)

            elif right_candidate == None:
                base = (left_candidate, base[1])

            elif not in_circumcircle(left_candidate, base[0], base[1], right_candidate):
                base = (base[0], right_candidate)

            else:
                base = (left_candidate, base[1])

            edges.append(base)  # Add the LR-edge

        return edges

    return build(points)


from matplotlib import pyplot as plt
import random
from scipy.spatial import Delaunay
import numpy as np

# Example usage:
if __name__ == "__main__":

    for i in range(100):
        random.seed(i)
        points = [(random.random(), random.random()) for _ in range(10)]
        points_np = np.array(points)

        # Divide and conquer Delaunay triangulation
        edges = delaunay(points)
        dac_edges = set(tuple(sorted(edge)) for edge in edges)

        # Scipy Delaunay implementation
        tri = Delaunay(points_np)
        scipy_edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted((tuple(points_np[simplex[i]]), tuple(points_np[simplex[(i + 1) % 3]]))))
                scipy_edges.add(edge)

        # Compare results
        print("Edges match:", dac_edges == scipy_edges)

        # Plot results
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot custom Delaunay
        for edge in edges:
            p1, p2 = edge
            xs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            ax[0].plot(xs, ys, "bo-")
        ax[0].set_title("Divide and Conquer Delaunay Triangulation")
        ax[0].set_aspect("equal")

        # Plot scipy Delaunay
        for edge in scipy_edges:
            p1, p2 = edge
            xs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            ax[1].plot(xs, ys, "ro-")
        ax[1].set_title("Scipy Delaunay Triangulation")
        ax[1].set_aspect("equal")

        plt.show()
