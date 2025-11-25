from setuptools import setup
from glob import glob
import os

package_name = 'rl_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],  # the Python package directory: src/rl_nav/rl_nav
    data_files=[
        # all paths here must be **relative**
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='PPO training on TurtleBot3 in Gazebo',
    license='MIT',
    entry_points={
        'console_scripts': [
            'train_ppo = rl_nav.train_ppo:main',  # <-- what ros2 run will call
            'ppo_controller_node = rl_nav.ppo_controller_node:main',
        ],
    },
)
