from rl_nav.constants import item_size_coordinates, item_mass, colors

def generate_item(item_name, color):
    inertia_x = (item_mass / 12.0) * (item_size_coordinates[1]**2 + item_size_coordinates[2]**2)
    inertia_y = (item_mass / 12.0) * (item_size_coordinates[0]**2 + item_size_coordinates[2]**2)
    inertia_z = (item_mass / 12.0) * (item_size_coordinates[0]**2 + item_size_coordinates[1]**2)
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{item_name}">
    <static>false</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>{item_mass}</mass>
        <inertia>
          <ixx>{inertia_x}</ixx>
          <iyy>{inertia_y}</iyy>
          <izz>{inertia_z}</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{item_size_coordinates[0]} {item_size_coordinates[1]} {item_size_coordinates[2]}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{item_size_coordinates[0]} {item_size_coordinates[1]} {item_size_coordinates[2]}</size></box></geometry>
        <material>
          <ambient>{color[0]} {color[1]} {color[2]} {color[3]}</ambient>
          <diffuse>{color[0]} {color[1]} {color[2]} {color[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
    return sdf


def get_item_color(task):
    return colors.get(task, colors['A'])
