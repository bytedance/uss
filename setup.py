from setuptools import find_packages, setup


setup(
    name='casa',
    version='0.0.1',
    description='Computational auditory scene anaysis (CASA) for universal source separation.',
    author='Qiuqiang Kong',
    author_email='qiuqiangkong@gmail.com',
    license='Apache2.0',
    packages=find_packages(),
    # package_data={'': ["LICENSE"]},
    # package_data={'casa': ["aa.txt"]},
    # data_files=["LICENSE"],
    include_package_data=True,
    # package_dir={"": "casa"},
    entry_points={
        'console_scripts': ['casa=casa.tmp:add'],
        # 'console_scripts': ['casa=casa.inference:separate'],
        # 'console_scripts': ['casa=casa.inference'],
        
    },
)