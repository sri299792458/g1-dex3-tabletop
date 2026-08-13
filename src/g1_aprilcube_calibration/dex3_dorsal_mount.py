"""Nominal geometry for the two-hole Dex3 dorsal ArUco mount.

The public Unitree Dex3 mesh contains the outside palm envelope but omits the
two physical dorsal holes.  The defaults in this module therefore distinguish
between two kinds of geometry:

* dimensions fixed by the printable design and the official palm mesh; and
* the dimensioned hole pattern from Unitree's Dex3-1 user manual.

All printable dimensions are millimetres.  Homogeneous transforms use metres.
The plate-local frame has ``+x`` across the hand, ``+y`` toward the fingers,
and ``+z`` from the printed marker face into the palm.  Consequently the
visible marker normal is plate ``-z``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

ARUCO_DICTIONARY_NAME = "DICT_6X6_50"
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_6X6_50


@dataclass(frozen=True, slots=True)
class Dex3DorsalMountSpec:
    """Dimensions for the direct, two-screw marker carrier.

    The hole spacing, depth, centerline, and longitudinal position come from
    the official dimensioned mounting-hole drawing.  The dorsal surface
    coordinate comes from ray intersection with Unitree's palm STL.
    """

    marker_id: int = 4
    marker_size_mm: float = 40.0
    marker_bits: int = 6
    border_bits: int = 1
    quiet_zone_mm: float = 5.0
    mounting_tab_depth_mm: float = 10.0
    mounting_tab_width_mm: float = 30.0
    plate_thickness_mm: float = 4.0
    plate_corner_radius_mm: float = 3.0
    black_inlay_depth_mm: float = 0.6

    # Official manual callout: 2 x M3, 3 mm deep, on 15.00 mm centers.  The
    # print-first coupon catches printer hole shrinkage before the full marker.
    hole_spacing_mm: float = 15.0
    thread_depth_mm: float = 3.0
    hole_clearance_diameter_mm: float = 3.4
    countersink_major_diameter_mm: float = 6.0
    countersink_depth_mm: float = 1.7
    hole_line_from_distal_edge_mm: float = 5.0
    minimum_tab_ligament_mm: float = 2.0
    nominal_screw_length_mm: float = 8.0

    # Three-point registration against the shallowly curved dorsal shell.
    screw_boss_diameter_mm: float = 7.0
    screw_boss_height_mm: float = 1.5
    wrist_pad_diameter_mm: float = 6.0
    wrist_pad_height_mm: float = 2.57
    wrist_pad_from_proximal_edge_mm: float = 20.0

    # Nominal mounting pose in either Dex3 palm-link frame.  The drawing places
    # the hole line 66.70 mm from the wrist-side datum and centered laterally.
    # The palm STL is x=[0, 86.7] mm with finger bases near x=77.7 mm, fixing
    # the drawing's wrist-to-finger direction as palm +x.
    palm_hole_midpoint_x_mm: float = 66.70
    palm_hole_midpoint_z_mm: float = 0.0
    palm_dorsal_surface_y_at_holes_mm: float = -21.33

    # Least-squares tangent slopes dy/dx from the official palm STL.  The
    # first is fitted over the two M3 boss footprints; the second is fitted
    # over the 6 mm wrist-pad footprint.  A positive mounting slope pitches
    # the plate away from the shell toward the wrist, which is why the former
    # 2.0 mm wrist pad did not make physical contact.
    palm_mounting_surface_dy_dx: float = 0.019947
    palm_wrist_surface_dy_dx: float = -0.01485
    palm_dorsal_surface_y_at_wrist_pad_mm: float = -20.95771

    @property
    def total_marker_cells(self) -> int:
        return self.marker_bits + 2 * self.border_bits

    @property
    def marker_cell_size_mm(self) -> float:
        return self.marker_size_mm / self.total_marker_cells

    @property
    def plate_size_mm(self) -> float:
        """Width and optical-square length before the mounting tab."""
        return self.marker_size_mm + 2.0 * self.quiet_zone_mm

    @property
    def plate_length_mm(self) -> float:
        return self.plate_size_mm + self.mounting_tab_depth_mm

    @property
    def mounting_pitch_deg(self) -> float:
        """Nominal pitch imposed by the local M3 mounting surface."""
        return float(np.degrees(np.arctan(self.palm_mounting_surface_dy_dx)))

    @property
    def wrist_pad_face_slope_dz_dy(self) -> float:
        """Pad-face slope in plate coordinates that matches the wrist shell."""
        mounting = self.palm_mounting_surface_dy_dx
        wrist = self.palm_wrist_surface_dy_dx
        return (wrist - mounting) / (1.0 + wrist * mounting)

    @property
    def wrist_pad_face_pitch_deg(self) -> float:
        return float(np.degrees(np.arctan(self.wrist_pad_face_slope_dz_dy)))

    @property
    def marker_face_y_in_palm_mm(self) -> float:
        """Nominal palm ``y`` coordinate of the marker center."""
        tangent_y = self.palm_mounting_surface_dy_dx / np.sqrt(
            1.0 + self.palm_mounting_surface_dy_dx**2
        )
        inward_y = 1.0 / np.sqrt(1.0 + self.palm_mounting_surface_dy_dx**2)
        marker_from_holes = self.plate_size_mm / 2.0 - (
            self.plate_length_mm - self.hole_line_from_distal_edge_mm
        )
        return (
            self.palm_dorsal_surface_y_at_holes_mm
            + marker_from_holes * tangent_y
            - (self.plate_thickness_mm + self.screw_boss_height_mm) * inward_y
        )

    @property
    def nominal_thread_engagement_mm(self) -> float:
        """M3 thread engagement with the specified countersunk screw."""
        return self.nominal_screw_length_mm - (self.plate_thickness_mm + self.screw_boss_height_mm)

    @property
    def nominal_bottoming_clearance_mm(self) -> float:
        return self.thread_depth_mm - self.nominal_thread_engagement_mm


DEFAULT_DEX3_DORSAL_MOUNT_SPEC = Dex3DorsalMountSpec()


def marker_grid(
    spec: Dex3DorsalMountSpec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC,
) -> np.ndarray:
    """Return the complete binary marker grid; zero is black and one is white."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
    pixels = spec.total_marker_cells
    image = cv2.aruco.generateImageMarker(
        dictionary,
        spec.marker_id,
        pixels,
        borderBits=spec.border_bits,
    )
    return (image > 0).astype(np.uint8)


def hole_centers_plate_mm(
    spec: Dex3DorsalMountSpec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC,
) -> np.ndarray:
    """Return the two hole centers in the printable plate-local frame."""
    center_x = spec.plate_size_mm / 2.0
    y = spec.plate_length_mm - spec.hole_line_from_distal_edge_mm
    return np.asarray(
        [
            [center_x - spec.hole_spacing_mm / 2.0, y, 0.0],
            [center_x + spec.hole_spacing_mm / 2.0, y, 0.0],
        ],
        dtype=np.float64,
    )


def palm_T_plate_mm(
    spec: Dex3DorsalMountSpec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC,
    *,
    side: str = "right",
) -> np.ndarray:
    """Map printable plate coordinates to the palm-link frame in millimetres.

    The plate follows the tangent of the mounting surface at the two M3 holes.
    Plate ``+x`` is across the hand, ``+y`` points toward the fingers, and
    ``+z`` points inward from the visible marker face.
    """
    if side not in {"left", "right"}:
        raise ValueError("Dex3 mount side must be left or right")
    # Unitree's official left palm is the right palm reflected across local Y.
    # The dorsal surface is therefore -Y on the right and +Y on the left.  To
    # keep the printable plate's +Y direction fingerward and its +Z direction
    # inward, its across-hand +X axis is +palm-Z on the right and -palm-Z on
    # the left.
    dorsal_sign = -1.0 if side == "right" else 1.0
    slope = -dorsal_sign * spec.palm_mounting_surface_dy_dx
    scale = np.sqrt(1.0 + slope**2)
    across = np.asarray([0.0, 0.0, -dorsal_sign])
    toward_fingers = np.asarray([1.0, slope, 0.0]) / scale
    inward = np.cross(across, toward_fingers)

    rotation = np.column_stack((across, toward_fingers, inward))
    hole_center_plate = np.asarray(
        [
            spec.plate_size_mm / 2.0,
            hole_centers_plate_mm(spec)[:, 1].mean(),
            spec.plate_thickness_mm + spec.screw_boss_height_mm,
        ]
    )
    hole_center_palm = np.asarray(
        [
            spec.palm_hole_midpoint_x_mm,
            dorsal_sign * abs(spec.palm_dorsal_surface_y_at_holes_mm),
            spec.palm_hole_midpoint_z_mm,
        ]
    )

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = hole_center_palm - rotation @ hole_center_plate
    return transform


def palm_T_marker_face(
    spec: Dex3DorsalMountSpec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC,
    *,
    side: str = "right",
) -> np.ndarray:
    """Return the nominal palm-link to visible-marker-center transform.

    Marker axes are defined as follows at the nominal mounting pitch:

    * ``+x``: across the hand;
    * ``+y``: along the plate toward the wrist; and
    * ``+z``: outward from the dorsal marker face.

    This is a proper right-handed optical target frame.  The transform is an
    initial value until the physical hole midpoint is measured or the fixed
    hand-to-marker pose is estimated by calibration.
    """
    palm_T_plate = palm_T_plate_mm(spec, side=side)
    plate_T_marker = np.eye(4)
    plate_T_marker[:3, :3] = np.diag([1.0, -1.0, -1.0])
    plate_T_marker[:3, 3] = np.asarray([spec.plate_size_mm / 2.0, spec.plate_size_mm / 2.0, 0.0])
    transform = palm_T_plate @ plate_T_marker
    transform[:3, 3] /= 1000.0
    return transform


def validate_mount_spec(
    spec: Dex3DorsalMountSpec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC,
) -> None:
    """Reject geometry that would weaken the plate or corrupt the target."""
    if isinstance(spec.marker_id, bool) or not isinstance(spec.marker_id, int):
        raise TypeError("marker ID must be an integer")
    if not 0 <= spec.marker_id < 50:
        raise ValueError("DICT_6X6_50 marker ID must lie within [0, 49]")
    positive = (
        spec.marker_size_mm,
        spec.quiet_zone_mm,
        spec.mounting_tab_depth_mm,
        spec.mounting_tab_width_mm,
        spec.plate_thickness_mm,
        spec.black_inlay_depth_mm,
        spec.hole_spacing_mm,
        spec.thread_depth_mm,
        spec.hole_clearance_diameter_mm,
        spec.countersink_major_diameter_mm,
        spec.countersink_depth_mm,
        spec.minimum_tab_ligament_mm,
        spec.nominal_screw_length_mm,
        spec.screw_boss_diameter_mm,
        spec.screw_boss_height_mm,
        spec.wrist_pad_diameter_mm,
        spec.wrist_pad_height_mm,
    )
    if any(value <= 0.0 for value in positive):
        raise ValueError("all mount dimensions must be positive")
    if spec.black_inlay_depth_mm >= spec.plate_thickness_mm:
        raise ValueError("black inlay must leave a continuous structural skin")
    if spec.countersink_depth_mm >= spec.plate_thickness_mm:
        raise ValueError("countersink must not consume the structural plate")
    minimum_tab_width = spec.hole_spacing_mm + spec.countersink_major_diameter_mm + 4.0
    if spec.mounting_tab_width_mm < minimum_tab_width:
        raise ValueError("mounting tab is too narrow around the two screw heads")
    holes = hole_centers_plate_mm(spec)
    radius = spec.countersink_major_diameter_mm / 2.0
    if np.any(holes[:, 0] < radius) or np.any(holes[:, 0] > spec.plate_size_mm - radius):
        raise ValueError("mounting hole head lies outside the carrier")
    if np.any(holes[:, 1] + radius > spec.plate_length_mm - spec.minimum_tab_ligament_mm):
        raise ValueError("insufficient distal ligament around mounting heads")
    if np.any(holes[:, 1] - radius < spec.plate_size_mm + spec.minimum_tab_ligament_mm):
        raise ValueError("insufficient marker-side ligament around mounting heads")
    if spec.nominal_thread_engagement_mm < 2.5:
        raise ValueError("M3 thread engagement must be at least 2.5 mm")
    if spec.nominal_bottoming_clearance_mm < 0.4:
        raise ValueError("blind-hole bottoming clearance must be at least 0.4 mm")


def mount_manifest(
    spec: Dex3DorsalMountSpec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC,
    *,
    side: str = "right",
) -> dict[str, object]:
    """Return a serializable record of the design and its evidence status."""
    return {
        "schema_version": 3,
        "hand_side": side,
        "units": "millimetres unless a transform is explicitly in metres",
        "sources": {
            "official_dimensioned_drawing": (
                "https://marketing.unitree.com/article/zh/Dex3-1/User_Manual.html"
            ),
            "official_urdf_and_mesh": ("unitree_ros/robots/dexterous_hand_description/dex3_1"),
        },
        "target": {
            "dictionary": ARUCO_DICTIONARY_NAME,
            "id": spec.marker_id,
            "active_size_mm": spec.marker_size_mm,
            "quiet_zone_mm": spec.quiet_zone_mm,
        },
        "spec": asdict(spec),
        "derived_geometry": {
            "mounting_pitch_deg": spec.mounting_pitch_deg,
            "wrist_pad_face_pitch_deg": spec.wrist_pad_face_pitch_deg,
        },
        "plate_hole_centers_mm": hole_centers_plate_mm(spec).tolist(),
        "fastener_stack": {
            "nominal_screw": "M3 x 8 mm ISO 10642",
            "printed_face_to_shell_mm": (spec.plate_thickness_mm + spec.screw_boss_height_mm),
            "effective_screw_stack_mm": (spec.plate_thickness_mm + spec.screw_boss_height_mm),
            "nominal_thread_engagement_mm": spec.nominal_thread_engagement_mm,
            "blind_hole_depth_mm": spec.thread_depth_mm,
            "nominal_bottoming_clearance_mm": spec.nominal_bottoming_clearance_mm,
        },
        "nominal_palm_T_marker_face_m": palm_T_marker_face(spec, side=side).tolist(),
        "evidence_status": {
            "marker_dictionary_and_id": (
                "decoded directly from the original GraspGen-X project video"
            ),
            "palm_outer_surface": "official Unitree Dex3 palm STL",
            "hole_thread_and_spacing": (
                "official Dex3-1 user-manual drawing: 2 x M3, 3 mm deep, 15.00 mm center spacing"
            ),
            "absolute_hole_pose_in_palm_link": (
                "manual drawing fixes x=66.70 mm and z=0; palm STL ray "
                "intersection fixes the local dorsal surface y=-21.33 mm"
            ),
            "plate_pitch_and_wrist_pad": (
                "least-squares tangent fits over the M3 boss and wrist-pad "
                "footprints in the official palm STL; physically confirmed "
                "by the first print leaving the former 2.0 mm pad unseated"
            ),
        },
    }


validate_mount_spec()
