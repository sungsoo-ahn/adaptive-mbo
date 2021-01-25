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
            "csm=design_baselines.csm.experiments:cli",
            "online=design_baselines.online.experiments:cli",
            "gradient-ascent=design_baselines.gradient_ascent.experiments:cli",
            "gan=design_baselines.gan.experiments:cli",
            "mins=design_baselines.mins.experiments:cli",
            "cbas=design_baselines.cbas.experiments:cli",
            "cma-es=design_baselines.cma_es.experiments:cli",
            "bo-qei=design_baselines.bo_qei.experiments:cli",
            "reinforce=design_baselines.reinforce.experiments:cli",
            "autofocus=design_baselines.autofocus.experiments:cli",
            "energy=design_baselines.energy.experiments:cli",
            "dual-cem=design_baselines.dual_cem.experiments:cli",
            "ssl=design_baselines.ssl.experiments:cli",
            "rwr=design_baselines.rwr.experiments:cli",
            "sgld=design_baselines.sgld.experiments:cli",
            "diffusion=design_baselines.diffusion.experiments:cli",
            "rollout=design_baselines.rollout.experiments:cli",
            "continual=design_baselines.continual.experiments:cli",
            "cga=design_baselines.cga.experiments:cli",
            "meta=design_baselines.meta.experiments:cli",
            "rebm=design_baselines.rebm.experiments:cli",
            "population=design_baselines.population.experiments:cli",
            "clip=design_baselines.clip.experiments:cli",
            "entmin=design_baselines.entmin.experiments:cli",
            "letgo=design_baselines.letgo.experiments:cli",
        )
    },
)
