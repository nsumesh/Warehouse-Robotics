from setuptools import setup
from glob import glob
import os

package_name = 'rl_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name], 
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),('share/' + package_name, ['package.xml']),(os.path.join('share', package_name, 'launch'), glob('launch/*.py')),],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nikhil Sumesh',
    maintainer_email='nsumesh@umd.edu',
    description='Virtual Sorting (Pickup and Dropoff) using PPO',
    license='MIT',
    entry_points={
        'console_scripts': [
            'train_ppo = rl_nav.train_ppo:main',
            'sorting_node = rl_nav.sorting_node:main',
            'eval_policy = rl_nav.eval_policy:main',
        ],
    },
)
