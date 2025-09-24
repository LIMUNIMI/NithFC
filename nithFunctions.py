import numpy as np
from udpFunctions import send_generic_udp_message


SENSORNAME = "NITHwebcamWrapper"
SENSORVERSION = "0.1.0"
OPCODE = "OPR"
MOUTH_APERTURE_MAX = str(0.6)
EYES_APERTURE_MAX = str(0.6)


def send_data(pitch, yaw, roll, mouth_aperture, left_eye_aperture, right_eye_aperture):
    message = (
        "$"
        + SENSORNAME
        + "-"
        + SENSORVERSION
        + "|"
        + OPCODE
        + "|"
        + "head_pos_pitch="
        + str(np.round(pitch, 2))
        + "&head_pos_yaw="
        + str(np.round(yaw, 2))
        + "&head_pos_roll="
        + str(np.round(roll, 2))
        + "&mouth_ape="
        + str(np.round(mouth_aperture, 2))
        # + MOUTH_APERTURE_MAX
        + "&eyeLeft_ape="
        + str(np.round(left_eye_aperture, 2))
        # + EYES_APERTURE_MAX
        + "&eyeRight_ape="
        + str(np.round(right_eye_aperture, 2))
        # + EYES_APERTURE_MAX
        + "&whistle_intensity=[50/75/100]"  
        + "^"
    )
    send_generic_udp_message(message)
