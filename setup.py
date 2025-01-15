from setuptools import setup, find_packages

# Read the requirements.txt file
with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="options",  # Your project name
    version="0.1.0",  # Initial version
    description="A package for pricing and analyzing exotic options",
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    packages=find_packages(),  # Automatically find all packages
    install_requires=install_requires,  # Use dependencies from requirements.txt
    python_requires=">=3.7",  # Minimum Python version
)
