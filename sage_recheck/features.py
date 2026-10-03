"""Geometric feature engine (ported from StudyBuddyAgent, single-frame adapted).

Ports compute_head_features, compute_shoulder_features, compute_body_features,
compute_two_shoulder_near, compute_two_ear_near from StudyBuddyAgent's
detect_sitting_posture_rule.py.  Adds compute_posture_features (torso angle,
standing detection) and compute_visibility_features (occlusion estimation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sage_recheck.geometry import calculate_angle, point_line_ratio

# COCO 17 keypoint indices
NOSE_IDX = "0"
SHOULDER_IDX = ["5", "6"]
HIP_IDX = ["11", "12"]
KNEE_IDX = ["13", "14"]
ANKLE_IDX = ["15", "16"]
CALCUL_CONFIDENCE = 0.3


@dataclass
class KeypointData:
    """17 COCO keypoints parsed from raw dict."""
    nose: Tuple
    left_eye: Tuple
    right_eye: Tuple
    left_ear: Tuple
    right_ear: Tuple
    left_shoulder: Tuple
    right_shoulder: Tuple
    left_elbow: Tuple
    right_elbow: Tuple
    left_wrist: Tuple
    right_wrist: Tuple
    left_hip: Tuple
    right_hip: Tuple
    left_knee: Tuple
    right_knee: Tuple
    left_ankle: Tuple
    right_ankle: Tuple
    has_nose: bool
    has_left_eye: bool
    has_right_eye: bool
    has_left_ear: bool
    has_right_ear: bool
    has_left_shoulder: bool
    has_right_shoulder: bool
    has_left_elbow: bool
    has_right_elbow: bool
    has_left_wrist: bool
    has_right_wrist: bool
    has_left_hip: bool
    has_right_hip: bool
    has_left_knee: bool
    has_right_knee: bool
    has_left_ankle: bool
    has_right_ankle: bool
    has_clear_left_shoulder: bool
    has_clear_right_shoulder: bool
    has_more_clear_left_shoulder: bool
    has_more_clear_right_shoulder: bool
    has_both_ears: bool
    has_both_shoulders: bool
    has_both_hips: bool


@dataclass
class FeatureReport:
    """All geometric features for one person."""
    nose_ratio: Optional[float]
    ear_tilt: float
    neck_tilt: Optional[float]
    upper_head_condition: bool
    two_ear_near: bool
    two_shoulder_near: bool
    ear_tilt_unreliable: bool
    has_both_ears: bool
    has_both_shoulders: bool
    has_both_hips: bool
    shoulder_tilt: float
    chinrest_condition: bool
    torso_angle: Optional[float]
    is_standing: Optional[bool]
    nose_below_shoulders: bool
    visibility_score: float
    critical_kp_visible: bool
    # v5: which measurement produced ear_tilt -- "pose_ears" (COCO ear
    # keypoints), "face_eye_line" (106-pt face landmark eye centers, used
    # when the pose ear geometry is missing or unreliable), or None (no
    # usable angular measurement; ear_tilt is then the -1.0 sentinel).
    ear_tilt_source: Optional[str] = None


def get_valid_keypoints(kp_dict: Dict, conf: float = CALCUL_CONFIDENCE) -> Dict[str, List[float]]:
    return {k: v for k, v in (kp_dict or {}).items() if len(v) >= 3 and v[2] >= conf}


def calc_head_height(kp_valid: Dict) -> Optional[float]:
    """Head height = (shoulder avg Y - nose Y) * 2."""
    if NOSE_IDX not in kp_valid:
        return None
    nose_y = kp_valid[NOSE_IDX][1]
    shoulders = [kp_valid[i] for i in SHOULDER_IDX if i in kp_valid]
    if not shoulders:
        return None
    avg_y = sum(s[1] for s in shoulders) / len(shoulders)
    return (avg_y - nose_y) * 2


def parse_keypoints(kp_dict: Dict) -> KeypointData:
    default = (0.0, 0.0, 0.0)
    kp = {str(i): default for i in range(17)}
    for k, v in (kp_dict or {}).items():
        if str(k) in kp and len(v) >= 3:
            kp[str(k)] = tuple(v[:3])
    return KeypointData(
        nose=kp["0"], left_eye=kp["1"], right_eye=kp["2"],
        left_ear=kp["3"], right_ear=kp["4"],
        left_shoulder=kp["5"], right_shoulder=kp["6"],
        left_elbow=kp["7"], right_elbow=kp["8"],
        left_wrist=kp["9"], right_wrist=kp["10"],
        left_hip=kp["11"], right_hip=kp["12"],
        left_knee=kp["13"], right_knee=kp["14"],
        left_ankle=kp["15"], right_ankle=kp["16"],
        has_nose=kp["0"][2] > 0,
        has_left_eye=kp["1"][2] > 0,
        has_right_eye=kp["2"][2] > 0,
        has_left_ear=kp["3"][2] > 0,
        has_right_ear=kp["4"][2] > 0,
        has_left_shoulder=kp["5"][2] > 0,
        has_right_shoulder=kp["6"][2] > 0,
        has_left_elbow=kp["7"][2] > 0,
        has_right_elbow=kp["8"][2] > 0,
        has_left_wrist=kp["9"][2] > 0,
        has_right_wrist=kp["10"][2] > 0,
        has_left_hip=kp["11"][2] >= CALCUL_CONFIDENCE,
        has_right_hip=kp["12"][2] >= CALCUL_CONFIDENCE,
        has_left_knee=kp["13"][2] >= CALCUL_CONFIDENCE,
        has_right_knee=kp["14"][2] >= CALCUL_CONFIDENCE,
        has_left_ankle=kp["15"][2] >= CALCUL_CONFIDENCE,
        has_right_ankle=kp["16"][2] >= CALCUL_CONFIDENCE,
        has_clear_left_shoulder=kp["5"][2] > 0.5,
        has_clear_right_shoulder=kp["6"][2] > 0.5,
        has_more_clear_left_shoulder=kp["5"][2] > 0.7,
        has_more_clear_right_shoulder=kp["6"][2] > 0.7,
        has_both_ears=kp["3"][2] > 0 and kp["4"][2] > 0,
        has_both_shoulders=kp["5"][2] > 0 and kp["6"][2] > 0,
        has_both_hips=kp["11"][2] >= CALCUL_CONFIDENCE and kp["12"][2] >= CALCUL_CONFIDENCE,
    )


def compute_two_shoulder_near(kp: KeypointData) -> bool:
    if not (kp.has_left_shoulder and kp.has_right_shoulder and kp.has_nose):
        return False
    smax = max(kp.left_shoulder[0], kp.right_shoulder[0])
    smin = min(kp.left_shoulder[0], kp.right_shoulder[0])
    if kp.left_shoulder[0] >= kp.nose[0] and kp.right_shoulder[0] >= kp.nose[0]:
        if (smin - kp.nose[0]) > 0.4 * (smax - kp.nose[0]):
            return True
        if kp.has_left_ear and (smax - smin) < (kp.left_ear[0] - kp.nose[0]):
            return True
    if kp.left_shoulder[0] <= kp.nose[0] and kp.right_shoulder[0] <= kp.nose[0]:
        if (kp.nose[0] - smax) > 0.4 * (kp.nose[0] - smin):
            return True
        if kp.has_right_ear and (smax - smin) < (kp.nose[0] - kp.right_ear[0]):
            return True
    return False


def compute_two_ear_near(kp: KeypointData) -> bool:
    if not (kp.has_left_ear and kp.has_right_ear and kp.has_nose):
        return False
    emax = max(kp.left_ear[0], kp.right_ear[0])
    emin = min(kp.left_ear[0], kp.right_ear[0])
    if kp.left_ear[0] >= kp.nose[0] and kp.right_ear[0] >= kp.nose[0]:
        if (emin - kp.nose[0]) > 0.25 * (emax - kp.nose[0]):
            return True
    if kp.left_ear[0] <= kp.nose[0] and kp.right_ear[0] <= kp.nose[0]:
        if (kp.nose[0] - emax) > 0.25 * (kp.nose[0] - emin):
            return True
    return False


def compute_shoulder_features(kp: KeypointData, depth_info: Dict) -> Tuple[float, bool]:
    """Returns (corrected_shoulder_tilt, infer_shelter). Depth optional."""
    infer_shelter = False
    if not (kp.has_clear_left_shoulder and kp.has_clear_right_shoulder):
        return -1.0, False
    sh_vec = (kp.right_shoulder[0] - kp.left_shoulder[0],
              kp.right_shoulder[1] - kp.left_shoulder[1])
    shoulder_tilt = calculate_angle(sh_vec, (-1, 0))
    shoulder_tilt = (180 - shoulder_tilt) if shoulder_tilt > 90 else shoulder_tilt
    if not depth_info:
        return shoulder_tilt, False
    left_d = depth_info.get("left_depth")
    right_d = depth_info.get("right_depth")
    mid_d = depth_info.get("mid_depth")
    if left_d is None or right_d is None or mid_d is None:
        return shoulder_tilt, False
    mean_d = (left_d + right_d) / 2
    depth_gap = 2 * abs(mid_d - mean_d) / (mid_d + mean_d + 1e-6)
    if mid_d < mean_d and depth_gap > 0.15:
        infer_shelter = True
        if left_d < right_d and left_d > mid_d:
            left_d = abs(2 * mid_d - right_d)
        elif left_d > right_d and right_d > mid_d:
            right_d = abs(2 * mid_d - left_d)
    depth_diff = abs(left_d - right_d)
    mean_d = (left_d + right_d) / 2
    rel_depth_diff = depth_diff / (mean_d + 1e-6)
    left_closer = left_d > right_d
    right_closer = right_d > left_d
    left_lower = kp.left_shoulder[1] > kp.right_shoulder[1]
    right_lower = kp.right_shoulder[1] > kp.left_shoulder[1]
    coef = max(0, 1 - 4 * rel_depth_diff ** 2)
    if left_closer:
        coef = coef if left_lower else 1
    elif right_closer:
        coef = coef if right_lower else 1
    else:
        coef = 1
    return shoulder_tilt * coef, infer_shelter


# 106-point face landmark index ranges of the eye landmark groups
# (same layout the eyesclosed strategy uses for its EAR computation).
_FACE_RIGHT_EYE = range(33, 43)
_FACE_LEFT_EYE = range(87, 97)


def eye_line_tilt(face_lm: List) -> Optional[float]:
    """Head-roll angle (deg) from the 106-point face landmark eye centers.

    Fallback angular measurement for frames where the pose model's ear
    keypoints are missing or degenerate (head-down / profile views): the
    eye line still tracks head roll there.  Unlike the ear line this
    measurement is NOT corrected for head turn (the nose_ratio correction
    needs ear geometry), so it reads systematically larger on turned heads
    -- the conservative direction (tilt claims stay supported, i.e. no
    false vetoes; contradiction needs a clearly level eye line).
    """
    try:
        re_x = sum(face_lm[i][0] for i in _FACE_RIGHT_EYE) / len(_FACE_RIGHT_EYE)
        re_y = sum(face_lm[i][1] for i in _FACE_RIGHT_EYE) / len(_FACE_RIGHT_EYE)
        le_x = sum(face_lm[i][0] for i in _FACE_LEFT_EYE) / len(_FACE_LEFT_EYE)
        le_y = sum(face_lm[i][1] for i in _FACE_LEFT_EYE) / len(_FACE_LEFT_EYE)
    except (IndexError, TypeError, ValueError):
        return None
    tilt = calculate_angle((re_x - le_x, re_y - le_y), (-1, 0))
    return (180 - tilt) if tilt > 90 else tilt


def compute_head_features(kp: KeypointData, shoulder_tilt: float,
                          infer_shelter: bool,
                          face_lm: Optional[List] = None) -> dict:
    """Ported from StudyBuddyAgent compute_head_features, LLM fields removed.

    face_lm: optional 106-point face landmarks (insightface) enabling the
    eye-line fallback when the pose ear geometry is missing or unreliable.
    """
    nose_ratio = 0.5
    corrected_ear_tilt = -1.0
    corrected_neck_tilt = -1.0
    upper_head_thresh = 0.5
    ear_tilt_unreliable = False
    upper_head_condition = True
    ear_tilt_source: Optional[str] = None
    tilt_head_basic_condition = kp.has_left_ear and kp.has_right_ear
    two_shoulder_near = compute_two_shoulder_near(kp)
    two_ear_near = compute_two_ear_near(kp)
    head_ear_state = (kp.has_left_ear and kp.has_right_ear and
                     (kp.right_ear[0] <= kp.left_ear[0] or two_ear_near))
    if head_ear_state:
        ear_vec = (kp.right_ear[0] - kp.left_ear[0],
                   kp.right_ear[1] - kp.left_ear[1])
        ear_tilt = calculate_angle(ear_vec, (-1, 0))
        corrected_ear_tilt = (180 - ear_tilt) if ear_tilt > 90 else ear_tilt
        if kp.has_nose:
            lower_head_coef = 1.0
            neck_tilt = -1.0
            nose_ratio = point_line_ratio(
                kp.right_ear[:2], kp.left_ear[:2], kp.nose[:2])
            turn_deg = abs(nose_ratio - 0.5)
            if nose_ratio > 0.9 or nose_ratio < 0.1:
                ear_tilt_unreliable = True
            corrected_ear_tilt = math.degrees(
                math.atan(math.tan(math.radians(corrected_ear_tilt)) *
                          math.cos(math.pi * turn_deg)))
            nose_between = (kp.left_ear[0] <= kp.nose[0] <= kp.right_ear[0] or
                           kp.left_ear[0] >= kp.nose[0] >= kp.right_ear[0])
            if nose_between:
                lower_head_coef = min((turn_deg + 0.5), 1.0)
                if nose_ratio > 0.75 and kp.has_right_eye:
                    if kp.right_ear[1] < kp.right_eye[1]:
                        ee_vec = (kp.right_ear[0] - kp.right_eye[0],
                                  kp.right_ear[1] - kp.right_eye[1])
                        neck_tilt = calculate_angle((-1, 0), ee_vec)
                    elif (kp.nose[1] < kp.right_ear[1] and
                          kp.nose[1] - kp.right_eye[1] > upper_head_thresh *
                          (kp.right_ear[1] - kp.right_eye[1])):
                        upper_head_condition = False
                if nose_ratio < 0.25 and kp.has_left_eye:
                    if kp.left_ear[1] < kp.left_eye[1]:
                        ee_vec = (kp.left_ear[0] - kp.left_eye[0],
                                  kp.left_ear[1] - kp.left_eye[1])
                        neck_tilt = calculate_angle((-1, 0), ee_vec)
                    elif (kp.nose[1] < kp.left_ear[1] and
                          kp.nose[1] - kp.left_eye[1] > upper_head_thresh *
                          (kp.left_ear[1] - kp.left_eye[1])):
                        upper_head_condition = False
                if 0.25 <= nose_ratio <= 0.75:
                    lower_head_coef = 0.75
                    rflag = kp.has_right_eye and kp.right_ear[1] < kp.right_eye[1]
                    lflag = kp.has_left_eye and kp.left_ear[1] < kp.left_eye[1]
                    if rflag and lflag:
                        r_vec = (kp.right_ear[0] - kp.right_eye[0],
                                  kp.right_ear[1] - kp.right_eye[1])
                        r_tilt = calculate_angle(ear_vec, r_vec)
                        r_tilt = (180 - r_tilt) if r_tilt > 90 else r_tilt
                        l_vec = (kp.left_ear[0] - kp.left_eye[0],
                                  kp.left_ear[1] - kp.left_eye[1])
                        l_tilt = calculate_angle(ear_vec, l_vec)
                        l_tilt = (180 - l_tilt) if l_tilt > 90 else l_tilt
                        neck_tilt = (l_tilt + r_tilt) / 2
                    elif rflag and not lflag:
                        ee_vec = (kp.right_ear[0] - kp.right_eye[0],
                                  kp.right_ear[1] - kp.right_eye[1])
                        neck_tilt = calculate_angle(ear_vec, ee_vec)
                    elif lflag and not rflag:
                        ee_vec = (kp.left_ear[0] - kp.left_eye[0],
                                  kp.left_ear[1] - kp.left_eye[1])
                        neck_tilt = calculate_angle(ear_vec, ee_vec)
                    mean_eye_y = (kp.left_eye[1] + kp.right_eye[1]) / 2
                    mean_ear_y = (kp.left_ear[1] + kp.right_ear[1]) / 2
                    if (kp.nose[1] < mean_ear_y and
                        kp.nose[1] - mean_eye_y > upper_head_thresh *
                        (mean_ear_y - mean_eye_y)):
                        upper_head_condition = False
            else:
                lower_head_coef = 1.0
                if kp.nose[0] > kp.left_ear[0] and kp.nose[0] > kp.right_ear[0]:
                    if kp.has_right_eye and kp.right_ear[1] < kp.right_eye[1]:
                        ee_vec = (kp.right_ear[0] - kp.right_eye[0],
                                  kp.right_ear[1] - kp.right_eye[1])
                        neck_tilt = calculate_angle((-1, 0), ee_vec)
                    elif (kp.has_right_eye and kp.nose[1] < kp.right_ear[1] and
                          kp.nose[1] - kp.right_eye[1] > upper_head_thresh *
                          (kp.right_ear[1] - kp.right_eye[1])):
                        upper_head_condition = False
                if kp.nose[0] < kp.left_ear[0] and kp.nose[0] < kp.right_ear[0]:
                    if kp.has_left_eye and kp.left_ear[1] < kp.left_eye[1]:
                        ee_vec = (kp.left_ear[0] - kp.left_eye[0],
                                  kp.left_ear[1] - kp.left_eye[1])
                        neck_tilt = calculate_angle((-1, 0), ee_vec)
                    elif (kp.has_left_eye and kp.nose[1] < kp.left_ear[1] and
                          kp.nose[1] - kp.left_eye[1] > upper_head_thresh *
                          (kp.left_ear[1] - kp.left_eye[1])):
                        upper_head_condition = False
            if neck_tilt != -1:
                neck_tilt = (180 - neck_tilt) if neck_tilt > 90 else neck_tilt
                corrected_neck_tilt = neck_tilt * lower_head_coef
        head_shoulder_state = (
            kp.has_clear_left_shoulder and kp.has_clear_right_shoulder and
            (kp.left_ear[1] - kp.right_ear[1]) *
            (kp.left_shoulder[1] - kp.right_shoulder[1]) > 0 and
            shoulder_tilt > 20 and
            (not infer_shelter or
             (kp.has_more_clear_left_shoulder and kp.has_more_clear_right_shoulder)))
        if not (kp.has_left_ear and kp.has_right_ear):
            corrected_ear_tilt = 0.0
        elif two_shoulder_near:
            pass
        elif head_shoulder_state:
            corrected_ear_tilt = max(
                0,
                corrected_ear_tilt - min(
                    (shoulder_tilt - 20) * 0.05 + 0.3, 1.0) * shoulder_tilt)
        ear_tilt_source = "pose_ears"
    # v6 face-primary roll: the ROI-cropped face eye line is geometrically
    # cleaner for head roll than the pose model's ear keypoints -- on the
    # classroom frames the pose ear line can read ~4 deg on a ~36 deg head
    # (ear keypoint placement error, even at high keypoint confidence).
    # The eye line is not corrected for head turn (conservative: reads
    # systematically larger, so tilt claims are only more supported and
    # contradiction needs a clearly level eye line).  Pose ears remain the
    # source when the face detector finds nothing.  neck_tilt /
    # upper_head_condition above still come from the pose ear geometry.
    if face_lm is not None and len(face_lm) >= 106:
        eye_tilt = eye_line_tilt(face_lm)
        if eye_tilt is not None:
            corrected_ear_tilt = eye_tilt
            ear_tilt_unreliable = False
            ear_tilt_source = "face_eye_line"
    elif head_ear_state:
        ear_tilt_source = "pose_ears"
    return {
        "nose_ratio": nose_ratio,
        # -1.0 sentinel when no angular measurement exists (degenerate ear
        # geometry AND no face fallback): consumers must treat ear_tilt < 0
        # as "unknown", never as a measured level head (the pre-v5 filler
        # 0.0 was silently read as a perfect contradiction signal).
        "ear_tilt": corrected_ear_tilt if corrected_ear_tilt >= 0 else -1.0,
        "neck_tilt": corrected_neck_tilt if corrected_neck_tilt >= 0 else None,
        "upper_head_condition": upper_head_condition,
        "two_ear_near": two_ear_near,
        "two_shoulder_near": two_shoulder_near,
        "ear_tilt_unreliable": ear_tilt_unreliable,
        "tilt_head_basic_condition": tilt_head_basic_condition,
        "ear_tilt_source": ear_tilt_source,
    }


def compute_body_features(kp: KeypointData) -> bool:
    """Chinrest basic condition (both hands raised to chin height)."""
    cond = True
    hs = (kp.has_left_shoulder and kp.has_right_shoulder and
          kp.has_left_wrist and kp.has_right_wrist and
          kp.has_left_elbow and kp.has_right_elbow)
    if hs:
        lh_down = kp.left_wrist[1] > kp.left_shoulder[1] + (kp.left_elbow[1] - kp.left_shoulder[1]) / 4
        rh_down = kp.right_wrist[1] > kp.right_shoulder[1] + (kp.right_elbow[1] - kp.right_shoulder[1]) / 4
        if lh_down or rh_down:
            cond = False
    if not (kp.has_left_wrist and kp.has_right_wrist):
        cond = False
    return cond


def compute_posture_features(kp: KeypointData,
                              head_height: Optional[float]) -> dict:
    """NEW: torso angle, standing detection, nose-below-shoulders."""
    torso_angle = None
    is_standing = None
    _POSTURE_HIP_CONF = 0.1
    if (kp.has_left_shoulder and kp.has_right_shoulder
            and kp.left_hip[2] >= _POSTURE_HIP_CONF
            and kp.right_hip[2] >= _POSTURE_HIP_CONF):
        sh_cy = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
        hip_cy = (kp.left_hip[1] + kp.right_hip[1]) / 2
        sh_cx = (kp.left_shoulder[0] + kp.right_shoulder[0]) / 2
        hip_cx = (kp.left_hip[0] + kp.right_hip[0]) / 2
        torso_vec = (hip_cx - sh_cx, hip_cy - sh_cy)
        torso_angle = calculate_angle(torso_vec, (0, 1))
        torso_angle = (180 - torso_angle) if torso_angle > 90 else torso_angle
        vgap = hip_cy - sh_cy
        if head_height and head_height > 0:
            if vgap < 0.3 * head_height:
                is_standing = True
            elif vgap > head_height:
                is_standing = False
        # Knee/ankle visibility does NOT imply standing -- a seated student
        # often has visible knees.  Only the torso vgap is a reliable signal.
    nose_below = False
    if kp.has_nose and kp.has_left_shoulder and kp.has_right_shoulder:
        nose_below = kp.nose[1] >= max(kp.left_shoulder[1], kp.right_shoulder[1])
    return {"torso_angle": torso_angle, "is_standing": is_standing,
            "nose_below_shoulders": nose_below}


def compute_visibility_features(kp_dict: Dict) -> dict:
    """NEW: visibility score and critical keypoint presence."""
    total = 17
    valid = 0
    for i in range(total):
        v = (kp_dict or {}).get(str(i))
        if v and len(v) >= 3 and v[2] >= CALCUL_CONFIDENCE:
            valid += 1
    score = valid / total if total > 0 else 0.0
    nose_vis = bool((kp_dict or {}).get(NOSE_IDX)) and kp_dict[NOSE_IDX][2] >= CALCUL_CONFIDENCE
    ls_vis = bool((kp_dict or {}).get("5")) and kp_dict["5"][2] >= CALCUL_CONFIDENCE
    rs_vis = bool((kp_dict or {}).get("6")) and kp_dict["6"][2] >= CALCUL_CONFIDENCE
    critical = nose_vis or ls_vis or rs_vis
    return {"visibility_score": score, "critical_kp_visible": critical}


class FeatureEngine:
    """Computes FeatureReport from keypoints + depth info."""

    def compute(self, kp_dict: Dict, kp_valid: Dict,
                head_height: Optional[float],
                depth_info: Optional[Dict] = None,
                face_lm: Optional[List] = None) -> Optional[FeatureReport]:
        if not kp_dict:
            return None
        kp = parse_keypoints(kp_dict)
        shoulder_tilt, infer_shelter = compute_shoulder_features(
            kp, depth_info or {})
        head = compute_head_features(kp, shoulder_tilt, infer_shelter,
                                     face_lm=face_lm)
        chinrest = compute_body_features(kp)
        posture = compute_posture_features(kp, head_height)
        vis = compute_visibility_features(kp_dict)
        return FeatureReport(
            nose_ratio=head["nose_ratio"],
            ear_tilt=head["ear_tilt"],
            neck_tilt=head["neck_tilt"],
            upper_head_condition=head["upper_head_condition"],
            two_ear_near=head["two_ear_near"],
            two_shoulder_near=head["two_shoulder_near"],
            ear_tilt_unreliable=head["ear_tilt_unreliable"],
            has_both_ears=kp.has_both_ears,
            has_both_shoulders=kp.has_both_shoulders,
            has_both_hips=kp.has_both_hips,
            shoulder_tilt=shoulder_tilt,
            chinrest_condition=chinrest,
            torso_angle=posture["torso_angle"],
            is_standing=posture["is_standing"],
            nose_below_shoulders=posture["nose_below_shoulders"],
            visibility_score=vis["visibility_score"],
            critical_kp_visible=vis["critical_kp_visible"],
            ear_tilt_source=head.get("ear_tilt_source"),
        )
