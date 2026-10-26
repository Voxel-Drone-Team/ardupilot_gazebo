#!/usr/bin/env python3
"""Writes the dock's AprilTag textures and model.sdf. Run after editing TAGS.

Model frame: origin on the floor at the centre of the bay footprint, +X out
of the opening, +Z up. Seen from the opening, +Y is to the right.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEXTURES = os.path.join(HERE, "materials", "textures")

FAMILY = cv2.aruco.DICT_APRILTAG_36h11
CELL_PX = 64

DEPTH = 1.5           # x, opening to back wall
WIDTH = 2.0           # y
WALL = 0.05
BACK_HEIGHT = 3.0
BAY_HEIGHT = 1.3      # underside of the shelf
SHELF = 0.1
CEILING = 0.1         # the upper bay's roof, its top level with the back wall's
PAD = 0.04
UPPER_FLOOR = BAY_HEIGHT + SHELF
UPPER_HEIGHT = BACK_HEIGHT - CEILING - UPPER_FLOOR  # shelf top to the ceiling's underside

BACK_FACE_X = -DEPTH / 2 + WALL

# (id, size of the black square in metres, surface, centre (along y, z) or
# (x, y) on the pad). Clusters follow the reference layout: a large centre tag
# for range, four small ones around it for close-in pose. The drone finds the
# dock from stereo_perception/config/dock_layout.yaml, which must match.
TAGS = [
    (11, 0.40, "back", (0.0, 2.25)),
    (13, 0.16, "back", (0.0, 2.25 + 0.42)),
    (12, 0.16, "back", (0.0, 2.25 - 0.42)),
    (14, 0.16, "back", (-0.42, 2.25)),
    (15, 0.16, "back", (0.42, 2.25)),
    (21, 0.24, "back", (0.0, 0.72)),
    (23, 0.10, "back", (0.0, 0.72 + 0.27)),
    (22, 0.10, "back", (0.0, 0.72 - 0.27)),
    (24, 0.10, "back", (-0.27, 0.72)),
    (25, 0.10, "back", (0.27, 0.72)),
    (30, 0.40, "pad", (0.0, 0.0)),
]


def texture_name(tag_id):
    return f"tag36_11_{tag_id:05d}.png"


def write_texture(tag_id):
    dictionary = cv2.aruco.getPredefinedDictionary(FAMILY)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 8 * CELL_PX, borderBits=1)
    # One white cell all round: the quiet zone detectors need, part of the
    # tag36h11 layout but outside its black square.
    image = np.full((10 * CELL_PX, 10 * CELL_PX), 255, np.uint8)
    image[CELL_PX:-CELL_PX, CELL_PX:-CELL_PX] = marker
    cv2.imwrite(os.path.join(TEXTURES, texture_name(tag_id)), image)


def box(name, size, pose, colour, collide=True):
    sx, sy, sz = size
    x, y, z = pose
    material = f"""
        <material>
          <ambient>{colour}</ambient>
          <diffuse>{colour}</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>"""
    geometry = f"<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>"
    out = f"""
      <visual name="{name}_visual">
        <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
        {geometry}{material}
      </visual>"""
    if collide:
        out += f"""
      <collision name="{name}_collision">
        <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
        {geometry}
      </collision>"""
    return out


def tag_visual(tag_id, size, surface, centre):
    side = size * 10 / 8
    if surface == "back":
        y, z = centre
        # Facing +X, out of the opening, from where the unrotated texture
        # reads the right way round.
        pose = f"{BACK_FACE_X + 0.002:.4f} {y:.3f} {z:.3f} 0 0 0"
        dims = f"0.002 {side:.4f} {side:.4f}"
    else:
        x, y = centre
        pose = f"{x:.3f} {y:.3f} {PAD + 0.001:.4f} 0 0 0"
        dims = f"{side:.4f} {side:.4f} 0.002"
    return f"""
      <visual name="tag_{tag_id}">
        <pose>{pose}</pose>
        <geometry><box><size>{dims}</size></box></geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0 0 0 1</specular>
          <pbr>
            <metal>
              <albedo_map>model://voxel_dock/materials/textures/{texture_name(tag_id)}</albedo_map>
              <metalness>0</metalness>
              <roughness>1</roughness>
            </metal>
          </pbr>
        </material>
      </visual>"""


def light(name, z):
    return f"""
      <light name="{name}" type="point">
        <pose>0.3 0 {z:.2f} 0 0 0</pose>
        <diffuse>0.9 0.9 0.9 1</diffuse>
        <specular>0.1 0.1 0.1 1</specular>
        <attenuation>
          <range>6</range>
          <constant>0.4</constant>
          <linear>0.1</linear>
          <quadratic>0.05</quadratic>
        </attenuation>
        <cast_shadows>false</cast_shadows>
      </light>"""


def main():
    os.makedirs(TEXTURES, exist_ok=True)
    for tag_id, *_ in TAGS:
        write_texture(tag_id)

    grey = "0.55 0.55 0.55 1"
    white = "0.95 0.95 0.95 1"
    dark = "0.25 0.25 0.25 1"
    side_y = WIDTH / 2 - WALL / 2
    parts = [
        box("back_wall", (WALL, WIDTH, BACK_HEIGHT), (-DEPTH / 2 + WALL / 2, 0, BACK_HEIGHT / 2), grey),
        box("left_wall", (DEPTH, WALL, BAY_HEIGHT), (0, -side_y, BAY_HEIGHT / 2), grey),
        box("right_wall", (DEPTH, WALL, BAY_HEIGHT), (0, side_y, BAY_HEIGHT / 2), grey),
        box("shelf", (DEPTH, WIDTH, SHELF), (0, 0, BAY_HEIGHT + SHELF / 2), white),
        box("pad", (DEPTH - WALL, WIDTH - 2 * WALL, PAD), (WALL / 2, 0, PAD / 2), dark),
        # The upper bay, closed like the lower one: sides on the shelf, a roof.
        box("upper_left_wall", (DEPTH, WALL, UPPER_HEIGHT), (0, -side_y, UPPER_FLOOR + UPPER_HEIGHT / 2), grey),
        box("upper_right_wall", (DEPTH, WALL, UPPER_HEIGHT), (0, side_y, UPPER_FLOOR + UPPER_HEIGHT / 2), grey),
        box("ceiling", (DEPTH, WIDTH, CEILING), (0, 0, BACK_HEIGHT - CEILING / 2), white),
    ]
    parts += [tag_visual(*tag) for tag in TAGS]

    lights = "".join(
        light(name, z) for name, z in (("bay_light", BAY_HEIGHT - 0.1), ("upper_bay_light", BACK_HEIGHT - CEILING - 0.1))
    )
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- Generated by generate.py; edit that and rerun instead of this file. -->
  <model name="voxel_dock">
    <static>true</static>
    <link name="base_link">{''.join(parts)}{lights}
    </link>
  </model>
</sdf>
"""
    with open(os.path.join(HERE, "model.sdf"), "w") as f:
        f.write(sdf)


if __name__ == "__main__":
    main()
