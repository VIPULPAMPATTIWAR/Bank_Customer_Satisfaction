from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)-> List[str]:
    '''
    this function return list of requirements
    '''
    HYPEN_E_DOT = "-e ."
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='PROJECT',
    version='0.0.1',
    author='vipul pampattiwar',
    author_email='vipulpampattiwar@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)