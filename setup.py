from setuptools import setup, Extension
import numpy

include_dirs_numpy = [numpy.get_include()]


def get_version_number():
    main_ns = {}
    for line in open('core/__init__.py', 'r').readlines():
        if not(line.find('__version__')):
            exec(line, main_ns)
            return main_ns['__version__']




setup(name='ExciPy',
      version=get_version_number(),
      description='ExciPy module for exciton dynamics by KMC method',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Tolib Abdurakhmonov',
      url='https://github.com/fizikximik',
      author_email='abdurakhmonov.t.z@gmail.com',
      packages=['core',
                'core.analysis',
                'core.processes',
                'core.kmc',
                'core.tools'],
      install_requires=['numpy', 'ase', 'matplotlib'],
      license='MIT License')
