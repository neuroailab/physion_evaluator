from setuptools import setup, find_packages

# Setuptools setup configuration
setup(
    name="physion_evaluator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "matplotlib",
        "h5py",
        "numpy"
    ],
    entry_points={
        'console_scripts': [
            'physion_feature_extract = physion_evaluator.extract_features:main',
            'physion_train_readout = physion_evaluator.train_readout:main',
        ],
    },
)
