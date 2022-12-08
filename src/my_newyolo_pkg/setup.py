from setuptools import setup

package_name = 'my_newyolo_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vishvajit',
    maintainer_email='vishvajitjambuti003@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'new_yolo = my_newyolo_pkg.new_yolo:main',
            'new_yolo_two = my_newyolo_pkg.new_yolo2:main',
        ],
    },
)
