from setuptools import find_packages, setup

package_name = 'color_docking'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shubhang',
    maintainer_email='shubhang.srikoti@gmail.com',
    description='Color-based docking node',
    license='MIT',
    entry_points={
        'console_scripts': [
        	'color_docking_node = color_docking.color_docking_node:main'
        ],
    },
)
