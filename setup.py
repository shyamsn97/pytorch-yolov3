from setuptools import setup, find_packages

setup(
    name="pytorch-yolov3",
    version="2.0",
    url="https://github.com/nrsyed/pytorch-yolov3",
    author="Najam R Syed",
    author_email="najam.r.syed@gmail.com",
    license="MIT",
    packages=find_packages(include=["yolov3"]),
    install_requires=[
    ],
    entry_points={
        "console_scripts": ["yolov3 = yolov3.__main__:main"]
    },
)
