from setuptools import find_packages
from setuptools import setup


setup(
    name="design-baselines",
    description="Baselines for Model-Based Optimization",
    license="MIT",
    version="0.1",
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(),
    entry_points={
        "console_scripts": (
            "design-baselines=design_baselines.cli:cli",
            "csty=design_baselines.csty.experiments:cli",
            "ensemble=design_baselines.ensemble.experiments:cli",
            "energy=design_baselines.energy.experiments:cli",
            "distil=design_baselines.distil.experiments:cli",
            "pess=design_baselines.pess.experiments:cli",
            "localsearch=design_baselines.localsearch.experiments:cli",
            "gradpess=design_baselines.gradpess.experiments:cli",
            "fastgradpess=design_baselines.fastgradpess.experiments:cli",
        )
    },
)
