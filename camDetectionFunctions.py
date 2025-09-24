import math
import cv2
import numpy as np
from typing import NamedTuple

# from shapely.geometry import Polygon

class Point(NamedTuple):
    x: float
    y: float

# An hardcoded value to try to normalize eyebrow aperture ratio vs front nose ratio
eyebrow_nose_normalizer = 4.28

# Define the landmarks for the face and eyes
face_border_indices = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
left_eye_indices = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
right_eye_indices = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]

left_eyebrow_top = 296
left_eyebrow_base = 297
right_eyebrow_top = 66
right_eyebrow_base = 67
front_nose_top = 10
front_nose_bottom = 9
left_eyebrow_pairs = [(336, 338), (296, 297)]
right_eyebrow_pairs = [(66, 67), (107, 109)]

def get_distance(p1: Point, p2: Point) -> float:
    """
    Calculate the Euclidean distance between two points in a 2D plane.

    Parameters:
    p1 (Point): A point having coordinates (x, y).
    p2 (Point): Another point having coordinates (x, y).

    Returns:
    float: The Euclidean distance between point p1 and point p2.

    Assumes that both p1 and p2 are objects with 'x' and 'y' attributes,
    representing their respective coordinates on the 2D plane.

    Note:
    This function requires the math module to be imported.

    Example:
    >>> p1 = Point(1, 2)
    >>> p2 = Point(4, 6)
    >>> get_distance(p1, p2)
    5.0
    """
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def get_face_orientation(face_landmarks, img_w, img_h):
    # Initialize lists to store 2D and 3D points
    face_2d = []
    face_3d = []

    for idx, lm in enumerate(face_landmarks.landmark):
        # Non so perché fa 'sta cosa
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Get 2D coordinates
            face_2d.append([x, y])

            # Get 3D coordinates
            face_3d.append([x, y, lm.z])

    # Convert to NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Camera matrix
    focal_length = 1 * img_w

    cam_matrix = np.array(
        [
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1],
        ]
    )

    # Distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    # This function call is where the magic happens. It uses the camera matrix and the distortion parameters to solve the Perspective-n-Point (PnP) problem. The PnP problem involves finding the position and orientation (rotation and translation) of a 3D object in space, given a set of 3D points on the object (face_3d), their corresponding 2D projections on the image (face_2d), and the camera parameters. In this case, it's likely trying to determine the position and orientation of a face relative to the camera.
    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Convert angles to degrees
    x = angles[0] * (360)
    y = angles[1] * (360)
    z = angles[2] * (360)

    return x, y, z

def get_eyebrows_ratio_2(face_landmarks) -> float:
    LLmtop = face_landmarks.landmark[338]
    LLmbot = face_landmarks.landmark[336]
    LLetop = face_landmarks.landmark[297]
    LLebot = face_landmarks.landmark[296]
    RRmtop = face_landmarks.landmark[109]
    RRmbot = face_landmarks.landmark[107]
    RRetop = face_landmarks.landmark[67]
    RRebot = face_landmarks.landmark[66]

    NOlandmarks = [
        face_landmarks.landmark[front_nose_top],
        face_landmarks.landmark[front_nose_bottom],
    ]

    # Calculate distances
    LLm = get_distance(LLmtop, LLmbot)
    LLe = get_distance(LLetop, LLebot)
    RRm = get_distance(RRmtop, RRmbot)
    RRe = get_distance(RRetop, RRebot)

    NOd = get_distance(NOlandmarks[0], NOlandmarks[1])

    # distances averages
    LL = (LLm + LLe) / 2
    RR = (RRm + RRe) / 2

    Left_aperture = LL / NOd if NOd > 0 else 0
    Right_aperture = RR / NOd if NOd > 0 else 0

    average = Left_aperture + Right_aperture / 2

    return average

def get_left_eyebrow_aperture_ratio(face_landmarks) -> float:
    # Landmarks for left eyebrow
    LElandmarks = [
        face_landmarks.landmark[left_eyebrow_top],
        face_landmarks.landmark[left_eyebrow_base],
    ]
    NOlandmarks = [
        face_landmarks.landmark[front_nose_top],
        face_landmarks.landmark[front_nose_bottom],
    ]

    # Calculate distances
    LEd = get_distance(LElandmarks[0], LElandmarks[1])
    NOd = get_distance(NOlandmarks[0], NOlandmarks[1])

    # Calculate ratios
    LEratio = LEd / NOd if NOd > 0 else 0

    return LEratio

def get_right_eyebrow_aperture_ratio(face_landmarks) -> float:
    # Landmarks for right eyebrow
    RElandmarks = [
        face_landmarks.landmark[right_eyebrow_top],
        face_landmarks.landmark[right_eyebrow_base],
    ]
    NOlandmarks = [
        face_landmarks.landmark[front_nose_top],
        face_landmarks.landmark[front_nose_bottom],
    ]

    # Calculate distances
    REd = get_distance(RElandmarks[0], RElandmarks[1])
    NOd = get_distance(NOlandmarks[0], NOlandmarks[1])

    # Calculate ratios
    REratio = REd / NOd if NOd > 0 else 0

    return REratio

def get_eyebrows_aperture_average(face_landmarks) -> float:
    return (get_left_eyebrow_aperture_ratio(face_landmarks) + get_right_eyebrow_aperture_ratio(face_landmarks)) / 2

def get_eye_aperture_ratio_SEGMENTSMETHOD(face_landmarks):
    # Landmarks for eye width calculation
    LElandmarksW = [
        face_landmarks.landmark[33],
        face_landmarks.landmark[133],
    ]  # Corrected indices for left eye horizontal
    RElandmarksW = [
        face_landmarks.landmark[362],
        face_landmarks.landmark[263],
    ]  # Corrected indices for right eye horizontal

    # Landmarks for eye height calculation
    LElandmarksH = [
        face_landmarks.landmark[159],
        face_landmarks.landmark[145],
    ]  # Corrected indices for left eye vertical
    RElandmarksH = [
        face_landmarks.landmark[386],
        face_landmarks.landmark[374],
    ]  # Corrected indices for right eye vertical

    # Calculate distances
    LEw = get_distance(LElandmarksW[0], LElandmarksW[1])
    REw = get_distance(RElandmarksW[0], RElandmarksW[1])
    LEh = get_distance(LElandmarksH[0], LElandmarksH[1])
    REh = get_distance(RElandmarksH[0], RElandmarksH[1])

    # Calculate ratios
    LERatio = LEh / LEw if LEw > 0 else 0
    RERatio = REh / REw if REw > 0 else 0

    return (LERatio, RERatio)


def get_eye_aperture_ratio_TWOSEGMENTSMETHOD(face_landmarks):
    # Landmarks for left eye
    LElandmarksW = [
        face_landmarks.landmark[33],
        face_landmarks.landmark[133],
    ]
    LElandmarksH1 = [
        face_landmarks.landmark[159],
        face_landmarks.landmark[145],
    ]
    LElandmarksH2 = [
        face_landmarks.landmark[158],
        face_landmarks.landmark[153],
    ]

    # Landmarks for right eye
    RElandmarksW = [
        face_landmarks.landmark[362],
        face_landmarks.landmark[263],
    ]
    RElandmarksH1 = [
        face_landmarks.landmark[386],
        face_landmarks.landmark[374],
    ]
    RElandmarksH2 = [
        face_landmarks.landmark[385],
        face_landmarks.landmark[380],
    ]

    # Calculate distances
    LEw = get_distance(LElandmarksW[0], LElandmarksW[1])
    LEh1 = get_distance(LElandmarksH1[0], LElandmarksH1[1])
    LEh2 = get_distance(LElandmarksH2[0], LElandmarksH2[1])

    REw = get_distance(RElandmarksW[0], RElandmarksW[1])
    REh1 = get_distance(RElandmarksH1[0], RElandmarksH1[1])
    REh2 = get_distance(RElandmarksH2[0], RElandmarksH2[1])

    # Calculate ratios
    LERatio = (LEh1 + LEh2) / (2 * LEw) if LEw > 0 else 0
    RERatio = (REh1 + REh2) / (2 * REw) if REw > 0 else 0

    return (LERatio, RERatio)


# def get_eye_face_ratio_AREASMETHOD(face_landmarks):
#     # Get the coordinates from the landmarks
#     face_coords = [
#         (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
#         for i in face_border_indices
#     ]
#     left_eye_coords = [
#         (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
#         for i in left_eye_indices
#     ]
#     right_eye_coords = [
#         (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
#         for i in right_eye_indices
#     ]

#     # Create polygons
#     face_polygon = Polygon(face_coords)
#     left_eye_polygon = Polygon(left_eye_coords)
#     right_eye_polygon = Polygon(right_eye_coords)

#     # Calculate areas
#     face_area = face_polygon.area
#     left_eye_area = left_eye_polygon.area
#     right_eye_area = right_eye_polygon.area

#     # Calculate ratios
#     left_eye_ratio = left_eye_area / face_area if face_area > 0 else 0
#     right_eye_ratio = right_eye_area / face_area if face_area > 0 else 0

#     return (left_eye_ratio, right_eye_ratio)


def get_mouth_aperture_ratio(face_landmarks):
    # Landmarks for upper to lower lip distance
    MlandmarksH = [face_landmarks.landmark[13], face_landmarks.landmark[14]]

    # Landmarks for mouth width
    MlandmarksW = [face_landmarks.landmark[78], face_landmarks.landmark[366]]

    # Get distances
    mh = get_distance(MlandmarksH[0], MlandmarksH[1])
    mw = get_distance(MlandmarksW[0], MlandmarksW[1])

    # Calculate ratios
    Mratio = mh / mw if mw > 0 else 0

    return Mratio


def get_head_roll(face_landmarks):
    # ^^ Calculate the angle described by the landmarks 127 and 356
    p1 = Point(face_landmarks.landmark[127].x, face_landmarks.landmark[127].y)
    p2 = Point(face_landmarks.landmark[356].x, face_landmarks.landmark[356].y)
    angle = math.atan2(p2.y - p1.y, p2.x - p1.x) * 180 / math.pi
    return angle
