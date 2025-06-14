import sys
import os

# Add the project root to Python path so we can import app modules
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import the dashboard module
from app.dashboard import main

if __name__ == "__main__":
    main()