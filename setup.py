from setuptools import setup, find_packages

setup(
    version = '0.0.1',
    name = 'motion-planners-study',
    author = 'Maksim Surov',
    author_email = 'surov.m.o@gmail.com',
    install_requires = [
        'sympy',
        'numpy',
        'matplotlib',
        'scipy',
        'ipython',
        'PyQT6',
        'ipykernel',
        'casadi_bspline @ git+https://github.com/msurov/casadi_bspline'
        # 'casadi_bspline @ file:///home/msurov/sirius/casadi_bspline'
    ],
    package_dir = {
        '': 'src'
    },
    packages = find_packages(where='src'),
)
