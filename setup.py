from setuptools import setup, find_packages


setup(name='ame',
      version='0.1',
      description='analytic Mean Embeddings two sample test ',
      url='https://github.com/kacperChwialkowski/analyticMeanEmbeddings',
      author='Kacper Chwialkowski',
      author_email='kacper.chwialkowski@gmail.com',
      license='BSD3',
      packages=find_packages('.', exclude=["*tests*", "*.develop"]),
      zip_safe=False)
