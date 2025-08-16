from setuptools import find_packages, setup
from typing import List

# -e . refers to setup.py file

def get_requirements()->List[str]:
    """
    This function returns the list of requirements
    """
    requirement_lst: List[str]=[]
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.readlines()
            for line in requirements:
                requirement = line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement!='-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
        
    return requirement_lst

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="CGbhakt420",
    author_email="sanchit.kr007@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
            