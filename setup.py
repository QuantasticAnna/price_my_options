from setuptools import setup, find_packages

# Read the requirements.txt file
with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="price_my_options",  
    version="0.1.0",  
    description="A package for pricing and analyzing exotic options",
    author="Annabelle Soula", 
    author_email="annabelle.soula@hotmail.com", 
    packages=find_packages(),  # Automatically find all packages
    install_requires=install_requires,  # Use dependencies from requirements.txt
    python_requires=">=3.7",  # Minimum Python version
)
