# jaankaup-engine examples. 

##

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
