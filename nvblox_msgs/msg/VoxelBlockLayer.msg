std_msgs/Header header

# Block size is the physical size (in meters) of a block side
float32 block_size_m
# Voxel size is the physical size (in meters) of a voxel side
float32 voxel_size_m
# Block indices are the 3D indices of the blocks; to get the origin of the
# block, simply multiply its index by the block size.
# Note that we consider a block's origin to be the low-side corner of
# the low-side voxel.
Index3D[] block_indices
# Blocks contains the actual mesh within the block, including triangles,
# vertex positions, and vertex colors.
VoxelBlock[] blocks
# Whether to clear the entire previous map. This is set to true when
# the *entire* map rather than only the pieces that changed are published.
bool clear
# Type identifier for the layer
int32 layer_type

# How the blocks are encoded. Receivers must reject encodings they do not know
# rather than reading empty centers as an empty map.
uint8 ENCODING_CENTERS=0
uint8 ENCODING_OCCUPANCY_BITMASK=1
uint8 encoding

# Increments once per published message. A receiver that sees a gap has missed
# an update and, when the publisher only sends changed blocks, cannot recover by
# waiting; it should call the publisher's resync service instead.
uint32 sequence

# True when the publisher sends a block only if its contents changed. A message
# with no blocks is then a normal "nothing moved" heartbeat rather than evidence
# that the mapping input died, and receivers must not treat quiet as starved.
bool delta_streaming
