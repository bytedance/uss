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
    install_requires=[
        "lightning==2.0.0",
        "h5py==3.8.0",
        "librosa==0.10.0.post2",
        "pandas==1.5.3",
        "panns_inference==0.1.0",
        "tensorboard==2.12.2",
        "einops==0.6.1",
    ],
    entry_points={
        'console_scripts': ['casa=casa.casa:main'],
    },
)