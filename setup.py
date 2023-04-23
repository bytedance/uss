from setuptools import find_packages, setup


setup(
    name='casa',
    version='0.0.1',
    description='Computational auditory scene anaysis (CASA) for universal source separation.',
    author='Qiuqiang Kong',
    author_email='qiuqiangkong@gmail.com',
    license='Apache2.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['casa=casa.inference'],
    },
)