from setuptools import find_packages, setup

setup(name='open3d_vis',
      version='0.1',
      install_requires=[
          'open3d>=0.16', 
          'natsort', 
          'cached-property', 
          'typeguard',
          'py-structs>=1.0.0',

          # Testing related
          'pytest',
      ],
      packages=find_packages(),
      entry_points={}
)
