from setuptools import setup
import os
from glob import glob

package_name = 'warehouse_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'worlds'),
            glob(os.path.join('worlds', '*.world'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user1',
    maintainer_email='user1@example.com',
    description='Warehouse simulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'warehouse_spawner = warehouse_sim.warehouse_spawner:main',
        ],
    },
)
