# <This script needs to be ran from within DarkMappy root directory>

# Install core and extra requirements
echo -ne 'Building Dependencies... \r'
pip install -q -r requirements/requirements-core.txt
echo -ne 'Building Dependencies... ########           (33%)\r'
pip install -q -r requirements/requirements-docs.txt
echo -ne 'Building Dependencies... ###########        (66%)\r'
pip install -q -r requirements/requirements-tests.txt
echo -ne 'Building Dependencies... ################   (100%)\r'
echo -ne '\n'

# Install Optimus-Primal (TODO: update PyPi and add to core reqs)
git clone git@github.com:astro-informatics/Optimus-Primal.git .optimus-primal
cd .optimus-primal
pip install .
cd ..

# Install specific converter for building tutorial documentation
conda install pandoc=1.19.2.1 -y

# Build the scattering emulator
pip install -e .