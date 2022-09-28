from distutils.core import setup

setup(name="nlp-core",
      version="0.0.1",
      description="Tools for NLP research",
      license="MIT",
      author="Henry Scheible",
      author_email="henry.scheible@gmail.com",
      url="https://github.com/henryscheible/nlp-core",
      install_requires=[
          'torch',
          'transformers[sentencepiece]',
          'datasets',
          'evaluate',
          'sklearn'
      ],
      packages="nlp-core")
