
Have you ever played a game like [Barony](https://store.steampowered.com/app/371970/Barony/) before? It's a lot of fun however it's extremely narrow in scope and not fun at all to play on your own. Verdigris, from a gameplay perspective, could be best described as a combination of the magic system of Barony with dynamic gameplay experience of modded Minecraft. More specifically the mods [Roguelike Dungeons](https://www.curseforge.com/minecraft/modpacks/roguelike-adventures-and-dungeons-2),  [The Twilight Forest](https://www.curseforge.com/minecraft/mc-mods/the-twilight-forest), and [Biomes O' Plenty](https://www.curseforge.com/minecraft/mc-mods/biomes-o-plenty)

Verdigris' initial gameplay loop is generally very simple:
1. Spawn into the world
2. Explore through the world to find a dungeon to loot (~30 / 45m)
3. Loot a dungeon (~1hr / 8hrs)
4. Return back to your homebase where you collect your loot together and make upgraded weapons and tools


While simple, this gameplay look should be sufficiently complex enough for an MVP and ripe for expansion with some simple possibilities listed below
- Creation of new Magic Items
- Exploring the world for resources
- Traveling to towns and cities to enhance magical abilities and/or sell items
- More dimensions / planes of existence

All gameplay will take place in a voxelized world that extends out to the horizon with minimal visual artifacts. Some type of global illumination must be an option for users with a high end graphics card, however the entire game should be runnable at minimum graphical settings at 30 FPS without global illumination on integrated graphics. 


MVP tasks:
1. Rendering of a single 512^3 chunk at >60 fps with direct and indirect lighting
2. Optimizing that single chunk so that it can be used in larger and larger LODs (for now just use Perlin or simplex noise) Target a visible area of 12k ^ 3 to start
3. Modify the player so that they can place / remove any voxel at any point in the world in a few different manners. Single, Sphere, Cube
4. ---- Graphics Checkpoint  ---- 
	- At this point you should have a fully dynamic world in which and voxels can be placed in and removed from in any order
5. Create a simple physics system which interacts with the player, the world, free voxels, and entities
6. Begin the creation of a simple magic system. Implement something like fireball and something that causes and explosion at a distance (Focus on visual, collision and damage support)   
7. Create simple enemies that fire these randomly into the world and optimize them so that you can spawn about ~1000 of them before any noticeable lag occurs
8. ---- Gameplay Checkpoint 1 ----
	- At this point it should be possible to walk around in the world, and shoot some simple spells at enemies, have them shoot some back at you, you can melee attack them and vice versa.
9. Begin working on changing world generation to be more interesting, add about ~10 simple biomes
10. Work on player interaction with this world, add tools and the ability to interact with the world with them
11. Add droppable voxels and an inventory system (bag of holding?)
12. ---- Gameplay Checkpoint 2 ----
	- At this point it should be possible to walk around and explore an open world, build anything you want, interact with some simple monsters
13. Add more monsters and 3 initial dungeons to explore
	1. Add a [Hydra's Lair](https://ftbwiki.org/File:Hydra_in_its_lair.png)
		- It's an open half dome, fireball is required to singe the heads
	2. Add a Wvyern's mountain
		- It's a mountain with a wyvern atop it. If you get too close it swoops down and tries to attack you, there is a loot hoard at the base
	3. Add a Catacombs 
		- It's underground, with a small entrance
1. Add simple towns and simple merchants, able to sell items you've made / looted and buy items
2. Add more biomes with a goal of around ~35 different ones
3. ---- MVP Checkpoint ---- 
	- At this point you should have a game. Show it off, ask others for their opinions. 



wandering trader