import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Adding pythonpath for import reasons
if 'PYTHONPATH' in os.environ:
    os.system(f'export PYTHONPATH=$PYTHONPATH:{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')
else:
    os.system(f'export PYTHONPATH={os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')
