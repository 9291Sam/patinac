Split Raster Chunk into 6
cull chunks that are outside of view

Make a UI

Make a spell
Misc:
    - everything refactor
    - scramble draw order dogfood!!!
    - Consume memory ordering bikeshedding?
    - Custom Linear Algebra library?
    - Put cache directly in renderer

pat | #c | scale| side  | total
3x3 | 9  | 1.0  | 512   | 1536
4r4 | 12 | 1.5  | 768   | 3072 
4r4 | 12 | 3.0  | 1536  | 6144 
4r4 | 12 | 6.0  | 3072  | 12288
4r4 | 12 | 12.0 | 6144  | 24576
4r4 | 12 | 12.0 | 12288 | 49152
4r4 | 12 | 12.0 | 24576 | 98304
-------------------------------
Total Chunks: 81
Rendered area: 98304^2 x 512 

greedy meshing
draw indriet
face level culling

jemealloc

create an mvp

make a really good readme
