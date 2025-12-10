"""
Constants shared across the rl_nav package.
"""

# Workspace limits - limited aisle workspace
X_MIN, X_MAX = -7.0, 2.0
Y_MIN, Y_MAX = -3.0, 3.0

# Docking bays - virtual sorting categories
# Each dock represents a different sorting category
DOCK_A = (-6.5, -2.0)  # Sorting category A
DOCK_B = (-6.5, 0.0)   # Sorting category B
DOCK_C = (-6.5, 2.0)   # Sorting category C

# Pickup zone
PICKUP = (-4.0, 0.0)

# Success and close zone radii (consistent across all methods)
SUCCESS_RADIUS = 0.7  # Distance threshold for successful docking
CLOSE_RADIUS = 1.5   # Distance threshold for intermediate close bonus

# Item properties
ITEM_SIZE = [0.2, 0.2, 0.3]  # 20cm x 20cm x 30cm box
ITEM_MASS = 0.5

# Item color mapping for visual distinction
ITEM_COLORS = {
    'A': [0.8, 0.2, 0.2, 1.0],  # Red
    'B': [0.2, 0.8, 0.2, 1.0],  # Green
    'C': [0.2, 0.2, 0.8, 1.0],  # Blue
}

# Action space for robot control
ACTIONS = [
    (0.12, 0.6),   # forward + turn left
    (0.15, 0.0),   # straight forward
    (0.12, -0.6),  # forward + turn right
    (0.00, 0.6),   # in-place left
    (0.00, -0.6),  # in-place right
]

# LiDAR and sensor parameters
MAX_RANGE = 3.5
NUM_SCAN_BINS = 24

# Docking parameters
DOCKING_TRANSITION_DISTANCE = 1.5  # meters
DOCKING_STABLE_DURATION = 3.0  # seconds
MAX_DOCKING_TIME = 120.0  # seconds
BLUE_MARKER_DETECTION_THRESHOLD = 800000  # pixels²
BLUE_MARKER_COMPLETE_THRESHOLD = 2000000  # pixels²
BLUE_MARKER_CENTERED_THRESHOLD = 50  # pixels

# Blue color range for marker detection (HSV)
BLUE_LOWER = [100, 150, 50]
BLUE_UPPER = [140, 255, 255]
