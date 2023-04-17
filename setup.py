from setuptools import find_packages, setup


def main():
    import huk_challenge as package

    setup(
        name='huk_challenge',
        version=package.__version__,
        author='Michael Gebauer',
        author_email='gebauerm23@gmail.com',
        packages=find_packages(),
        install_requires=package.install_requires,
        dependency_links=[""],
        python_requires=">=3.9",
        include_package_data=True,
        )


if __name__ == "__main__":
    main()
