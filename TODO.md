make your own linear algebra lib

use a nested lot of deques for allocating all of the bricks t

modify render cache so that Renderables can be created and implemented outside of the gfx crate

write a proper crash handler.
    Spawn all threads yourself, set panic hooks on them and make it generate proper crash hooks

what if you just completely ditch the idea of chunks at all and just have CPU allocated bricks?
on the cpu side you can just use some fancy 3d hash table and store brick positions

