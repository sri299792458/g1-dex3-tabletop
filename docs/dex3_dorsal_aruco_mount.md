# Dex3 two-hole dorsal ArUco mount

## Result

The mount is a **single 50 x 50 mm optical plate with a centered 30 x 10 mm
finger-side mounting tab**, fixed directly to the two holes in the rigid Dex3
palm shell. Its overall envelope is 50 x 60 mm. It carries a two-filament 40 mm
`DICT_6X6_50` marker, ID 4. There is no tape, intermediate bracket, snap joint,
or attachment to a moving finger.

This is the pitch-corrected second revision. The first physical print proved
that the former 2.0 mm wrist pad did not contact the hand. That design assumed
the plate was parallel to the palm-link plane. The local M3 mounting surface is
actually pitched by approximately 1.143 degrees. Revision 2 follows that pitch
and uses a 2.57 mm third datum on the flatter central shell region.

![Rendered mount on the official Dex3 geometry](../renders/dex3_dorsal_aruco_mount.png)

The printable H2D file is
[`dex3_dorsal_aruco_mount_multicolor_h2d.3mf`](../cad/dex3_dorsal_aruco_mount/dex3_dorsal_aruco_mount_multicolor_h2d.3mf).
Print the small
[`m3_15mm_fit_coupon.stl`](../cad/dex3_dorsal_aruco_mount/m3_15mm_fit_coupon.stl)
first.

## What is established, and what is inferred

The design does not pretend that the public Unitree model contains data that it
does not contain.

| Item | Status | Evidence |
|---|---|---|
| Target family | Established | The marker in the original GraspGen-X project video decodes as OpenCV `DICT_6X6_50`, ID 4, across multiple frames. |
| Target form | Established | Video frames show a flat, roughly 50 mm dorsal plate on the right Dex3. |
| Palm frame and outside surface | Established | Official Unitree `right_hand_palm_link.STL` and Dex3 URDF. The central dorsal region is shallowly curved: less than 0.9 mm departure from a plane over the proposed footprint. |
| Two physical holes | Established | Supplied photograph of the actual Dex3. |
| Hole pattern | **Official dimension** | Unitree's Dex3-1 user manual calls out `2 x M3`, 3 mm deep, on 15.00 mm centers. An M3 screw was also checked directly in the real hand. |
| Hole midpoint in `*_hand_palm_link` | **Official drawing + official mesh** | The manual places the pair centerline 66.70 mm from the wrist-side datum and on the hand centerline. The URDF/mesh establishes that as palm `[x, z] = [66.70, 0] mm`; ray intersection gives the dorsal shell at approximately `y = -21.33 mm`. |

The tiny coupon remains useful because FDM holes often print undersize. It now
checks printer clearance against an official 15.00 mm pattern rather than
testing an image-derived estimate.

The reproducible measurements are recorded in
[`graspgenx_dex3_mount_evidence.md`](../references/graspgenx_dex3_mount_evidence.md).
The raw downloaded video was analysis material and is not required by the CAD.
That evidence record also documents the search of Unitree's public CAD, model,
URDF, USD, historical mesh releases, and the image-based Dex3 user manual. The
URDF mesh omits the holes, but the manual supplies their missing dimensions.

## Geometry

| Feature | Dimension |
|---|---:|
| Carrier envelope | 50 x 60 x 4.0 mm |
| Optical square / centered mounting tab | 50 x 50 mm / 30 x 10 mm |
| Corner radius | 3.0 mm |
| Active ArUco square | 40.0 mm |
| Quiet zone | 5.0 mm on every side |
| Marker grid | 6 x 6 payload + one-cell black border |
| Cell size | 5.0 mm |
| Black inlay depth | 0.6 mm |
| Hole center spacing | 15.00 mm, official drawing |
| Hole pair centerline from wrist datum | 66.70 mm, official drawing |
| Blind-hole specification | 2 x M3, 3.0 mm deep |
| Through clearance | 3.4 mm |
| 90-degree countersink | 6.0 mm major diameter, 1.7 mm deep |
| Screw-line distance from finger-side tab edge | 5.0 mm |
| Material ligament around each 6 mm head, fore/aft | 2.0 mm / 2.0 mm |
| Local mounting pitch from palm STL | 1.143 degrees |
| Rear screw bosses | 7.0 mm diameter x 1.5 mm high |
| Rear stabilizing datum | 6.0 mm diameter x 2.57 mm center height |
| Stabilizing datum location | 20.0 mm from wrist-side optical edge |
| Stabilizing datum face correction | 1.994 degrees relative to plate |

The screw centers are on the separate 10 mm tab. Even the complete 6 mm heads
remain beyond the 50 mm optical square, so neither the black border nor its 5 mm
white quiet zone is interrupted. The line of screws sits next to the finger
bases and the complete marker extends toward the wrist. This keeps both metal
heads and the target away from the moving finger links.

The 10 mm tab depth is intentionally not reduced further. A centered 6 mm
countersink leaves 2 mm of PLA toward the optical plate and 2 mm toward the
outer edge. Reducing the overall length to 58.5 mm would save only 1.5 mm while
cutting the marker-side ligament to 0.5 mm, an undesirable stress concentration
next to a clamped countersunk screw.

The two annular screw bosses and one central circular pad form a three-point
seat. This is deliberate: the official shell is curved, so a nominally flat
back would rock or bend when tightened. Three small hard datums are the usual
fixture-design solution because they establish one plane without overconstraint.
The third datum is now only 35 mm from the screw line rather than 45 mm. It is
still wristward of the marker center, so the marker's center lies inside the
support triangle, but the pad no longer reaches onto the more strongly curved
wrist end of the shell.

The palm-STL tangent over the two screw-boss footprints has `dy/dx = 0.019947`.
The fitted tangent over the third pad is `dy/dx = -0.01485`. Those two slopes
set the installed plate pitch, the 2.57 mm pad height, and its angled contact
face. The nominal pad-center discrepancy from the STL surface is below 0.001
mm; printing tolerance, not the CAD equation, is now the limiting factor.

## Exact hardware

Use:

- 2 x **M3 x 8 mm, ISO 10642, 90-degree countersunk socket-head machine
  screws, black oxide**;
- one 2.0 mm hex key;
- white PLA and matte black PLA.

The printed stack from the flush head face to the hand shell remains 5.5 mm: 4.0 mm
plate plus a 1.5 mm rear boss. An M3 x 8 screw therefore enters by 2.5 mm,
leaving 0.5 mm before the conservative 3.0 mm blind-hole bottom. Do not
substitute M3 x 10: it can bottom in the blind hole. Do not use a washer; the
countersunk head is the intended plate datum.

The thread size has already been confirmed with an M3 screw. When assembling,
still start both screws by fingers only; this avoids cross-threading the shallow
blind holes.

## Print on the Bambu H2D

1. Print the fit coupon in any PLA using a 0.4 mm nozzle and 0.20 mm layers. It
   should take only a few minutes and checks printer tolerance and center
   spacing, not the already-confirmed thread designation.
2. Put two loose M3 screws through the coupon. Both screws must enter the hand
   holes without bending the coupon or pushing either screw sideways.
3. Open the multicolor 3MF as one object with two parts. Assign
   `carrier_white_PLA` to white and `aruco_black_PLA` to matte black.
4. Keep the generated orientation: the marker face is on the build plate and
   the three datum pads point upward. A smooth PEI plate gives the cleanest
   optical face. No support is required.
5. Use a 0.20 mm layer height, four walls, and at least five top/bottom layers.
   The 0.6 mm inlay occupies exactly three nominal layers.
6. Do not rescale either part and do not separate their origins.

Separate white and black STLs are included for slicers that do not preserve the
3MF material assignment.

## Assembly

1. Power the robot off and make sure no finger can move.
2. Place the plate with the screw line toward the finger bases and the long
   portion of the plate extending toward the wrist.
3. Start both M3 x 8 screws by hand before tightening either one.
4. Tighten alternately with the short end of the 2.0 mm key. Stop as soon as all
   three rear pads are seated and the plate no longer rocks. This is a marker,
   not a structural payload bracket; high torque only risks the Dex3 threads.
5. Verify that the plate clears both proximal finger links throughout their
   slow, powered-off manual range before commanding the hand.
6. In the detector, configure `DICT_6X6_50`, ID 4, and a marker side length of
   exactly `0.040` metres.

After assembly, take one close photograph square to the marker. The detector
must recover ID 4 with all four corners. The screw heads are outside the optical
square and should never enter the detected quadrilateral.

## Marker transform

The generated manifest records this nominal target frame:

```text
marker +x = [ 0,          0,          1 ] in the palm frame
marker +y = [-0.999801,  -0.019943,   0 ] toward the wrist
marker +z = [ 0.019943,  -0.999801,   0 ] outward from the marker face

palm translation to marker center = [0.03681565, -0.02742720, 0.0] m
```

The longitudinal and lateral coordinates come from Unitree's official
dimensioned drawing, not photograph registration. The outward coordinate and
1.143-degree plate pitch use the official palm mesh at the documented hole
positions. This is the transform to use for the planned camera-only solve after
the physical seating and fixed-transform validation checks. Keep the optional
joint camera/marker solve available as a diagnostic comparison rather than the
production result.

The CAD is parametric. After measuring a different center spacing, regenerate
it with:

```bash
uv run python tools/generate_dex3_dorsal_aruco_mount.py \
  --hole-spacing-mm MEASURED_SPACING
```

The generated plate occupies a local 50 x 60 x 6.675 mm print envelope before
its palm-frame mounting transform is applied. Add the resulting pitched object
to any motion-planning model before using the marked hand near the torso, table,
or the other arm.
