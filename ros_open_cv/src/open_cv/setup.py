from setuptools import find_packages, setup

package_name = 'open_cv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='martin_ngn',
    maintainer_email='martin_ngn@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = open_cv.cameraPublisher:main',
            'subscribe_image = open_cv.subscribeImage:main',
            'sam_subscribe = open_cv.samSubscribe:main',
            'image_sub = open_cv.subIm:main',
        ],
    },
)
