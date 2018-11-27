from distutils.core import setup

setup(
    name='mlproject',
    version='0.1',
    author='Leon Sixt',
    author_email='github@leon-sixt.de',
    packages=['mlproject'],
    install_requires=[
        'sacred',
        'tqdm',
        'pillow',
        'torchvision',
        'tensorboardX',
        'gridfs',
        'pymongo'
    ]
)
