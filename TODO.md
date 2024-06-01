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
Total Chunks: 81
Rendered area: 98304^2 x 512 

greedy meshing
draw indriet
face level culling

jemealloc

create an mvp

make a really good readme

redo all of the channel things because youre making them all piecemeal lmfao

create user stories and tasks lol
code review fix imports and exports in lib.rs


if you want someone to help comment the codebase



once you need to do culling you can split things up into indirect calls
use "vertex pulling" to get the u32 of the voxel's data (face)
makeyour faces u32s 9 bits each xyz pos and then a normal

from there look into the brick map and get the material data and stuff


in each indirect call you store the chunk's ptr which holds a material buffer


indirect calls per chunk and per direction cpu side culling should be dead simple (gpu culling is also an option lol)


remove that "rt" look by using some type of logarithmic falloff for lighting i.e
1 power gets it to ike 0.9 brightness 
10 gets it to 0.98
100 to 0.999

this prevents things getting washed out


add how many calls are rendererd / how many objects are registererd 

TODO: figure out tab off clicking

occlussion culling (replace the 0.5 with a 0.0 when off axis)

cap tick time
fix input lag