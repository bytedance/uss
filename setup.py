from setuptools import find_packages, setup


setup(
    name='uss',
    version='0.0.4',
    description='Universal source separation (USS) with weakly labelled data.',
    author='Qiuqiang Kong',
    author_email='qiuqiangkong@gmail.com',
    license='Apache2.0',
    packages=find_packages(),
    url="https://github.com/bytedance/uss",
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "lightning>=2.0.0",
        "panns_inference>=0.1.0",
        "transformers",
        "h5py",
        "librosa>=0.10.0.post2",
        "pandas",
        "tensorboard",
        "einops",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': ['uss=uss.uss_inference:main'],
    },
)
