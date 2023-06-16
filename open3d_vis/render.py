from numbers import Number
import numpy as np
import open3d as o3d

import torch

from .conversion import to_numpy
from camera_geometry.transforms import batch_transform_points, join_rt

from beartype import beartype


def line_set(points, edges, colors=None, color=None):
  points, edges = to_numpy(points), to_numpy(edges)

  lines = o3d.geometry.LineSet(
      points=o3d.utility.Vector3dVector(points), 
      lines=o3d.utility.Vector2iVector(edges))

  if colors is not None:
    lines.colors = o3d.utility.Vector3dVector(to_numpy(colors))
  elif color is not None:
    lines.paint_uniform_color(color)

  return lines


def triangle_mesh(vertices, triangles, vertex_colors=None):
  vertices, triangles = to_numpy(vertices), to_numpy(triangles)

  mesh = o3d.geometry.TriangleMesh(
      vertices=o3d.utility.Vector3dVector(vertices), 
      triangles=o3d.utility.Vector3iVector(triangles))

  if vertex_colors is not None:
    mesh.vertex_colors = o3d.utility.Vector3dVector(to_numpy(vertex_colors))

  return mesh


def segments(points1, points2, colors=None, color=None):
  points1, points2 = to_numpy(points1), to_numpy(points2)

  assert points1.shape == points2.shape
  n = points1.shape[0]

  return line_set(
    points = np.concatenate([points1, points2]),
    edges = np.stack([np.arange(n), np.arange(n, 2 * n)], axis=1),
    colors = colors,
    color = color
  )



def point_cloud(points, color=[0, 1, 0], colors=None):
  points = to_numpy(points)
  pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
      
  if colors is None:
    pcd = pcd.paint_uniform_color(color)
  else:
    pcd.colors = o3d.utility.Vector3dVector(to_numpy(colors))
  return pcd

def line_sequence(points, color=[0, 0, 1]):
  points = to_numpy(points)

  idx = np.arange(len(points) - 1)
  edges = np.stack([idx, idx + 1], axis=1)

  # print(points.shape, edges.shape)
  return line_set(points, edges).paint_uniform_color(color)


def line_loops(points, color=[0, 0, 1]):
  points = to_numpy(points)

  n, m, _ = points.shape
  idx = np.arange(m)
  edges = np.stack([idx, idx + 1], axis=1) % m
  
  loops = np.arange(n) * m
  edges = edges.reshape(1, *edges.shape) + loops.reshape(n, 1, 1)
  
  return line_set(
      points.reshape(-1, 3), 
      edges.reshape(-1, 2),
    ).paint_uniform_color(color)




def box(min_bound, max_bound, color=[0, 0, 1], wire=False):
  min_bound, max_bound = to_numpy(min_bound), to_numpy(max_bound)

  x, y, z = max_bound - min_bound
  box = o3d.geometry.TriangleMesh.create_box(x, y, z)

  box.vertices = o3d.utility.Vector3dVector(np.asarray(box.vertices) + min_bound)

  if wire is True:
    lines = o3d.geometry.LineSet.create_from_triangle_mesh(box)
    return lines.paint_uniform_color(color)
  else:
    return box.paint_uniform_color(color)


def sphere(pos, radius=1.0, color=[0, 0, 1]):
  pos = to_numpy(pos)

  sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
  sphere.vertices = o3d.utility.Vector3dVector(np.asarray(sphere.vertices) + pos)
  sphere.compute_vertex_normals()

  sphere.paint_uniform_color(color)
  return sphere




def instance_mesh(mesh, transforms, instance_colors=None):
  if instance_colors is not None:
    instance_colors = to_numpy(instance_colors)

  transforms = to_numpy(transforms)

  mesh_triangles = np.asarray(mesh.triangles)
  mesh_vertices = np.asarray(mesh.vertices)
  transforms = to_numpy(transforms)

  n = transforms.shape[0]
  tri_offsets = (np.arange(n) * mesh_vertices.shape[0]).reshape(n, 1, 1)

  vertices = batch_transform_points(transforms, mesh_vertices)
  triangles = mesh_triangles + tri_offsets

  vertex_colors = None
  if mesh.has_vertex_colors():
    vertex_colors = np.asarray(mesh.vertex_colors).reshape(1, -1, 3)
    vertex_colors = np.repeat(vertex_colors, n, axis=0).reshape(-1, 3)


  mesh = triangle_mesh(vertices.reshape(-1, 3), triangles.reshape(-1, 3), vertex_colors)

  if instance_colors is not None:
    instance_colors = to_numpy(instance_colors).reshape(-1, 1, 3)
    instance_colors = np.broadcast_to(instance_colors, (n, mesh_vertices.shape[0], 3))

    mesh.vertex_colors = o3d.utility.Vector3dVector(instance_colors.reshape(-1, 3)) 
  return mesh


def instance_mesh_at(mesh, offsets, scales, instance_colors=None):
  offsets = to_numpy(offsets)
  scales = to_numpy(scales)


  if isinstance(scales, Number):
    scales = np.full((offsets.shape[0],), scales,
                     dtype=np.float32).reshape(-1, 1, 1)
  else:
    assert scales.shape == offsets.shape 
    scales = to_numpy(scales).reshape(-1, 1, 3)

  m = np.eye(3).reshape(1, 3, 3) * scales

  transforms = join_rt(m, offsets)
  return instance_mesh(mesh, transforms, instance_colors=instance_colors)




def instance_lines(line_inst, transforms, instance_colors=None):

  inp_lines = np.asarray(line_inst.lines)
  inp_points = np.asarray(line_inst.points)
  transforms = to_numpy(transforms)

  n = transforms.shape[0]
  line_offsets = (np.arange(n) * inp_points.shape[0]).reshape(n, 1, 1)

  points = batch_transform_points(transforms, inp_points)
  lines = inp_lines + line_offsets

  colors = None
  if line_inst.has_colors():
    colors = np.asarray(line_inst.colors).reshape(1, -1, 3)
    colors = np.repeat(colors, n, axis=0).reshape(-1, 3)


  combined = line_set(points.reshape(-1, 3), lines.reshape(-1, 2), colors)

  if instance_colors is not None:
    instance_colors = to_numpy(instance_colors).reshape(-1, 1, 3)
    instance_colors = np.broadcast_to(instance_colors, (n, inp_lines.shape[0], 3))

    combined.colors = o3d.utility.Vector3dVector(instance_colors.reshape(-1, 3)) 
  return combined


def concat_mesh(meshes):

  vertices  = [np.asarray(mesh.vertices, dtype=np.float32) for mesh in meshes]
  triangles = [np.asarray(mesh.triangles, dtype=np.int32) for mesh in meshes]

  vertex_colors = None
  if meshes[0].has_vertex_colors():
    vertex_colors=np.concatenate([np.asarray(mesh.vertex_colors, dtype=np.float32) for mesh in meshes])

  offsets = 0
  for tri, verts in zip(triangles, vertices):
    tri += offsets
    offsets += verts.shape[0]

  return triangle_mesh(np.concatenate(vertices), np.concatenate(triangles), vertex_colors=vertex_colors)



def concat_lines(line_sets):

  points = [np.asarray(line.points, dtype=np.float32) for line in line_sets]
  lines  = [np.asarray(line.lines, dtype=np.int32) for line in line_sets]


  colors = None
  if line_sets[0].has_colors():
    colors = [np.asarray(line.colors, dtype=np.float32) for line in line_sets]

  offsets = 0
  for line, p in zip(lines, points):
    line += offsets
    offsets += p.shape[0]


  return line_set(
      np.concatenate(points), 
      np.concatenate(lines), 
      colors=np.concatenate(colors))




def spheres(points, radii, resolution=6, colors=None):
  sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=resolution)
  sphere.compute_vertex_normals()

  mesh = instance_mesh_at(sphere, points, radii, colors=colors)
  return mesh


def boxes(lower, upper, colors=None):

  box = o3d.geometry.TriangleMesh.create_box(1, 1, 1)

  mesh = instance_mesh_at(box, lower, upper-lower, instance_colors=colors)
  mesh.compute_triangle_normals()

  
  return mesh


def cloud(points, colors=None):
  points = to_numpy(points)

  pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

  if colors is not None:
    colors = to_numpy(colors)
    pcd.colors = o3d.utility.Vector3dVector(colors)

  return pcd



def vertex_dirs(points):
  d = points[1:] - points[:-1]
  d = d / np.linalg.norm(d)
  
  smooth = (d[1:] + d[:-1]) * 0.5
  dirs = np.concatenate([
    np.array(d[0:1]), smooth, np.array(d[-2:-1])
  ])

  return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def random_unit(dtype=np.float32):
 x =  np.random.randn(3).astype(dtype)
 return x / np.linalg.norm(x)


def make_tangent(d, n):
  t = np.cross(d, n)
  t /= np.linalg.norm(t, axis=-1, keepdims=True)
  return np.cross(t, d)

def gen_tangents(dirs, t):
  tangents = []

  for dir in dirs:
    t = make_tangent(dir, t)
    tangents.append(t)

  return np.stack(tangents)

@beartype
def unit_circle(n):
  a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
  return np.stack( [np.sin(a), np.cos(a)], axis=1)


def tube_vertices(points, radii, n=10):
  points = to_numpy(points)
  radii = to_numpy(radii)

  circle = unit_circle(n).astype(np.float32)

  dirs = vertex_dirs(points)
  t = gen_tangents(dirs, random_unit())

  b = np.stack([t, np.cross(t, dirs)], axis=1)
  b = b * radii.reshape(-1, 1, 1)

  return np.einsum('bdx,md->bmx', b, circle)\
     + points.reshape(points.shape[0], 1, 3)
  

@beartype
def tube_loops(points, radii, n:int=10):
  circles = tube_vertices(points, radii, n)
  return line_loops(circles)


def cylinder_triangles(m, n):

  tri1 = np.array([0, 1, 2])
  tri2 = np.array([2, 3, 0])

  v0 = np.arange(m)
  v1 = (v0 + 1) % m
  v2 = v1 + m
  v3 = v0 + m

  edges = np.stack([v0, v1, v2, v3], axis=1) 
 
  segments = np.arange(n - 1) * m
  edges = edges.reshape(1, *edges.shape) + segments.reshape(n - 1, 1, 1)

  edges = edges.reshape(-1, 4)
  return np.concatenate( [edges[:, tri1], edges[:, tri2]] )


def tube_mesh(points, radii, n=10):
  points = tube_vertices(points, radii, n)
  n, m, _ = points.shape
  indexes = cylinder_triangles(m, n)

  mesh = triangle_mesh(points.reshape(-1, 3), indexes)
  mesh.compute_vertex_normals()

  return mesh


