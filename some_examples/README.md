# jaankaup-engine examples. 

## The fast marching method

You can compile and run the project as follows. Beware, the project is under contruction.

$ cargo run --example fmm 

### Plan.

| FmmCell            |
| :---               |
| tag: u32           |
| value: f32         |
| update: u32        |
| misc: u32          |

**tag { FAR, BAND, KNOWN }**

**value is the generalized distance from the initial interface.**

**update: the previous update number.**

**misc: a bit mask that holds information for the cell block index, and for each direction (does the FmmCell has neighbor).**

|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10| 9| 8| 7| 6| 5| 4| 3| 2| 1| 0|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|bi|-z|+z|-y|+y|-x|+x| 

| FmmCellSync         |
| :---                |
| tag: u32            |
| value: f32          |
| queue_value: u32    |
| mem_location: u32   |
| block_index: u32    |

| FmmBlock              |
| :---                  |
| index: u32            |
| band_point_count: f32 |

* fmm_data : List<FmmCell>
* fmm_blocks : List<FmmBlock>
* active_fmm_blocks : List<FmmBlock>
* syncList : List<FmmSyncCell>
* isotropic_data: List<f32>

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

- [x] Phase 1 
- [x] Phase 2 
- [x] Phase 3 
- [ ] Phase 4 
- [ ] Phase 5 
- [ ] Phase 6 

## Curves

Test scene for space filling curves. The indices are now hard coded to [0..4096].

You can compile and run the project as follows

$ cargo run --example curves 

Shortkeys

* `a` Move left.
* `d` Move right.
* `w` Move forward.
* `s` Move backward.
* `c` Move down.
* `spacebar` Move up.
* `Key1` Hilbert index.
* `Key2` Rosenber--Strong.
* `Key3` "Cube slice" curve.
* `Key4` Morton code.
* `Key9` 64 thread mode. This can be changed in the source file: THREAD_COUNT = n.
* `Key0` Leave 64 thread mode.
* `l` increase arrow size.
* `k` decrease arrow size.
