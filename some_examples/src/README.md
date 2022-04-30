# The fast marching method

## Plan.

| FmmCell            |
| :---
| tag: u32           |
| value: f32         |

h

+---------------------+
| FmmCellSync         |
+---------------------+
|                     |
| * tag: u32          |
| * value: f32        |
| * queue_value: u32  |
| * mem_location: u32 |
| * block_index: u32  |
|                     |
+---------------------+

+-------------------------+
| FmmBlock                |
+-------------------------+
|                         |
| * index: u32            |
| * band_point_count: f32 |
|                         |
+-------------------------+

fmm_data : List<FmmCell>
fmm_blocks : List<FmmBlock>
active_fmm_blocks : List<FmmBlock>
syncList : List<FmmSyncCell>
isotropic_data: List<f32>

Fmm algorithm

   * Method 1 : all speed values are 1.0 (euclidic distance field)
   * Method 2 : from isotropic speed information (time arrival from the initial interface)

1. Create the initial fmm interface (initial known points)
   * this can be created from triangle meshes, distance/denstity function etc.
2. Create the initial band points
   * Add the possible band points to syncList. 
3. Calculate the band points values and merge the band points from syncList to fmm_data.

4. Update the fmm_blocks (those blocks that contain any band points are added to the active_fmm_blocks list.
5. For each fmm_block in fmm_blocks.
   * reduce (try to find the smalles band point and change the tag to KNOWN).
   * if a new KNOWN cell is found, update the neigbhorhood (find and add new BAND points to syncList)
6. Merge the syncList to fmm_data. Jump to 4 (update the fmm_blocks list). Is the active_fmm_blocks list is empty break loop. 
