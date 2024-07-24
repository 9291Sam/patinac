use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use gfx::glm;
use itertools::{iproduct, Itertools};
use rapier3d::dynamics::{IslandManager, RigidBodyBuilder, RigidBodyHandle, RigidBodySet};
use rapier3d::geometry::{ColliderBuilder, ColliderHandle, ColliderSet};
use voxel::{ChunkCoordinate, ChunkLocalPosition, CHUNK_EDGE_LEN_VOXELS};

pub struct VoxelWorld
{
    uuid: util::Uuid,

    critical_section: Mutex<CriticalSection>,
    pool:             Arc<voxel::ChunkPool>
}

struct CriticalSection
{
    chunks: HashMap<
        voxel::ChunkCoordinate,
        (voxel::Chunk, Vec<(voxel::ChunkLocalPosition, voxel::Voxel)>)
    >,
    current_colliders:       HashMap<WorldPosition, ColliderHandle>,
    current_player_position: glm::Vec3
}

#[derive(Clone, Copy, Hash, PartialEq, PartialOrd, Eq)]
pub struct WorldPosition(pub glm::I32Vec3);

fn world_position_to_chunk_position(
    WorldPosition(world_pos): WorldPosition
) -> (voxel::ChunkCoordinate, voxel::ChunkLocalPosition)
{
    (
        voxel::ChunkCoordinate(
            world_pos.map(|x| x.div_euclid(voxel::CHUNK_EDGE_LEN_VOXELS as i32))
        ),
        voxel::ChunkLocalPosition(
            world_pos
                .map(|x| x.rem_euclid(voxel::CHUNK_EDGE_LEN_VOXELS as i32))
                .try_cast()
                .unwrap()
        )
    )
}

fn world_position_from_chunk_positions(
    coordinate: voxel::ChunkCoordinate,
    pos: voxel::ChunkLocalPosition
) -> WorldPosition
{
    WorldPosition(coordinate.0 * CHUNK_EDGE_LEN_VOXELS as i32 + pos.0.cast())
}

impl VoxelWorld
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let this = Arc::new(VoxelWorld {
            uuid:             util::Uuid::new(),
            pool:             voxel::ChunkPool::new(game.clone()),
            critical_section: Mutex::new(CriticalSection {
                chunks:                  HashMap::new(),
                current_colliders:       HashMap::new(),
                current_player_position: glm::Vec3::zeros()
            })
        });

        game.register(this.clone());

        this
    }

    pub fn write_many_voxel(&self, voxels: impl IntoIterator<Item = (WorldPosition, voxel::Voxel)>)
    {
        let CriticalSection {
            chunks, ..
        } = &mut *self.critical_section.lock().unwrap();

        for (pos, voxel) in voxels
        {
            let (coordinate, local_pos) = world_position_to_chunk_position(pos);

            match chunks.entry(coordinate)
            {
                std::collections::hash_map::Entry::Occupied(mut e) =>
                {
                    e.get_mut().1.push((local_pos, voxel))
                }
                std::collections::hash_map::Entry::Vacant(e) =>
                {
                    e.insert((
                        self.pool.allocate_chunk(coordinate),
                        vec![(local_pos, voxel)]
                    ));
                }
            }
        }
    }

    pub fn flush_all_voxel_updates(&self)
    {
        let CriticalSection {
            chunks, ..
        } = &mut *self.critical_section.lock().unwrap();

        for (_, (chunk, voxels_to_flush)) in chunks.iter_mut()
        {
            self.pool.write_many_voxel(chunk, voxels_to_flush.drain(..));
        }
    }

    pub fn update_with_camera_position(&self, position: glm::Vec3)
    {
        self.critical_section
            .lock()
            .unwrap()
            .current_player_position = position;
    }

    pub fn write_lights(&self, lights: &[voxel::PointLight])
    {
        self.pool.write_lights(lights);
    }
}

impl game::EntityCastDepot for VoxelWorld
{
    fn as_self_managed(self: Arc<Self>) -> Option<Arc<dyn game::SelfManagedEntity>>
    {
        None
    }

    fn as_positionalable(&self) -> Option<&dyn game::Positionalable>
    {
        Some(self)
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        Some(self)
    }

    fn as_collideable(&self) -> Option<&dyn game::Collideable>
    {
        Some(self)
    }
}

impl game::Entity for VoxelWorld
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        std::borrow::Cow::Borrowed("VoxelWorld")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn tick(&self, _: &game::Game, _: game::TickTag) {}
}

impl game::Positionalable for VoxelWorld
{
    fn get_position(&self) -> glm::Vec3
    {
        glm::Vec3::new(0.0, 0.0, 0.0)
    }
}

impl game::Transformable for VoxelWorld
{
    fn get_transform(&self) -> gfx::Transform
    {
        gfx::Transform::new()
    }
}

impl game::Collideable for VoxelWorld
{
    fn init_collideable(
        &self
    ) -> (
        rapier3d::prelude::RigidBody,
        Vec<rapier3d::prelude::Collider>
    )
    {
        (RigidBodyBuilder::fixed().enabled(false).build(), Vec::new())
    }

    fn physics_tick(
        &self,
        _: &game::Game,
        _: glm::Vec3,
        _: RigidBodyHandle,
        collider_set: &mut ColliderSet,
        rigid_body_set: &mut RigidBodySet,
        islands: &mut IslandManager,
        _: game::TickTag
    )
    {
        let CriticalSection {
            chunks,
            current_colliders,
            current_player_position
        } = &mut *self.critical_section.lock().unwrap();

        let player_position: glm::I32Vec3 = current_player_position.map(|p| p.floor() as i32);

        let mut chunk_coordinate_view_positions_map: HashMap<
            ChunkCoordinate,
            Vec<ChunkLocalPosition>
        > = HashMap::new();

        iproduct!(
            (player_position.x - 3)..(player_position.x + 3),
            (player_position.y - 25)..(player_position.y + 25),
            (player_position.z - 3)..(player_position.z + 3),
        )
        .map(|(x, y, z)| WorldPosition(glm::I32Vec3::new(x, y, z)))
        .map(world_position_to_chunk_position)
        .for_each(|(coordinate, local_pos)| {
            match chunk_coordinate_view_positions_map.entry(coordinate)
            {
                std::collections::hash_map::Entry::Occupied(mut e) => e.get_mut().push(local_pos),
                std::collections::hash_map::Entry::Vacant(e) =>
                {
                    e.insert(vec![local_pos]);
                }
            }
        });

        let mut filled_positions_near_player: HashSet<WorldPosition> = HashSet::new();

        for (coordinate, positions) in chunk_coordinate_view_positions_map
        {
            if let Some((chunk, _)) = chunks.get(&coordinate)
            {
                let result = self
                    .pool
                    .read_many_voxel_occupied(chunk, positions.iter().cloned());

                for (chunk_local_pos, filled) in positions.iter().zip_eq(result)
                {
                    if filled
                    {
                        filled_positions_near_player.insert(world_position_from_chunk_positions(
                            coordinate,
                            *chunk_local_pos
                        ));
                    }
                }
            }
        }

        let mut colliders_to_cull: HashSet<(WorldPosition, ColliderHandle)> = HashSet::new();
        let mut colliders_to_add: HashSet<WorldPosition> = HashSet::new();

        for near_player in filled_positions_near_player.iter()
        {
            if !current_colliders.contains_key(near_player)
            {
                colliders_to_add.insert(*near_player);
            }
        }

        for (current_collider_position, handle) in current_colliders.iter()
        {
            if !filled_positions_near_player.contains(current_collider_position)
            {
                colliders_to_cull.insert((*current_collider_position, *handle));
            }
        }

        for (pos, handle) in colliders_to_cull
        {
            collider_set.remove(handle, islands, rigid_body_set, false);

            current_colliders.remove(&pos);
        }

        for c in colliders_to_add
        {
            let handle =
                collider_set.insert(ColliderBuilder::cuboid(0.5, 0.5, 0.5).translation(c.0.cast()));

            current_colliders.insert(c, handle);
        }
    }
}

// fix culling!
