"""
Item management utilities for generating SDF models and managing items.
"""
from rl_nav.constants import ITEM_SIZE, ITEM_MASS, ITEM_COLORS


def generate_item_sdf(item_name, color_rgba, size=None, mass=None):
    """
    Generate SDF XML for a sortable item box.
    
    Args:
        item_name: Name of the item model
        color_rgba: RGBA color values [r, g, b, a]
        size: Box size [x, y, z]. Defaults to ITEM_SIZE
        mass: Mass of the item. Defaults to ITEM_MASS
    
    Returns:
        str: SDF XML string
    """
    if size is None:
        size = ITEM_SIZE
    if mass is None:
        mass = ITEM_MASS
    
    # Calculate inertia (for box: I = m/12 * (h^2 + d^2))
    ixx = (mass / 12.0) * (size[1]**2 + size[2]**2)
    iyy = (mass / 12.0) * (size[0]**2 + size[2]**2)
    izz = (mass / 12.0) * (size[0]**2 + size[1]**2)
    
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{item_name}">
    <static>false</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx>
          <iyy>{iyy}</iyy>
          <izz>{izz}</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>
        <material>
          <ambient>{color_rgba[0]} {color_rgba[1]} {color_rgba[2]} {color_rgba[3]}</ambient>
          <diffuse>{color_rgba[0]} {color_rgba[1]} {color_rgba[2]} {color_rgba[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
    return sdf


def get_item_color(task_class):
    """
    Get color for a task class.
    
    Args:
        task_class: Task class ('A', 'B', or 'C')
    
    Returns:
        list: RGBA color values
    """
    return ITEM_COLORS.get(task_class, ITEM_COLORS['A'])
