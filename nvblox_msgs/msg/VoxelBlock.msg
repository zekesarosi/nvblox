# All of the constituent parts must either have the same length or be length 0.
geometry_msgs/Point32[] centers
std_msgs/ColorRGBA[] colors

# Packed alternative to centers/colors, used when the enclosing layer declares
# ENCODING_OCCUPANCY_BITMASK. A block is a fixed 8^3 grid at a known index, so
# binary occupancy is 512 bits; bit (i % 8) of occupancy[i / 8] is the voxel at
# linear index i = z + 8*y + 64*x, matching nvblox's own block layout. centers
# and colors are empty in that case.
uint8[] occupancy
