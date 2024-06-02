use std::borrow::Cow;
use std::sync::Arc;

use dot_vox::DotVoxData;
use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::{Rng, SeedableRng};
use util::AtomicF32;
use voxel2::{ChunkManager, WorldPosition};

use crate::instanced_indirect::InstancedIndirect;
use crate::recordables::flat_textured::FlatTextured;
use crate::recordables::lit_textured::LitTextured;

extern "C" {

    static DEMO_FLOAT_HEIGHT: AtomicF32;
}

#[derive(Debug)]
pub struct DemoScene
{
    _dm:               Arc<voxel2::ChunkManager>,
    _draws:            Vec<Arc<dyn gfx::Recordable>>,
    lit_textured_cube: Arc<LitTextured>,
    id:                util::Uuid
}

impl DemoScene
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let dm = ChunkManager::new(game.clone());
        let c_dm = dm.clone();
        let c_dm2 = dm.clone();

        let mut rng = rand::rngs::SmallRng::seed_from_u64(23879234789234);

        let draws = spiral::ChebyshevIterator::new(0, 0, 4)
            .map(|(x, z)| {
                [
                    LitTextured::new_cube(
                        game.clone(),
                        gfx::Transform {
                            translation: glm::Vec3::new(
                                x as f32 * 12.0 - 64.0,
                                164.0,
                                z as f32 * 12.0 + 64.0
                            ),
                            scale: glm::Vec3::repeat(4.0),
                            ..Default::default()
                        }
                    ) as Arc<dyn gfx::Recordable>,
                    FlatTextured::new_pentagon(
                        game.clone(),
                        gfx::Transform {
                            translation: glm::Vec3::new(
                                x as f32 * 12.0 - 64.0,
                                184.0,
                                z as f32 * 12.0 + 64.0
                            ),
                            scale: glm::Vec3::repeat(-16.0),
                            ..Default::default()
                        }
                    ) as Arc<dyn gfx::Recordable>
                ]
            })
            .flatten()
            .chain([InstancedIndirect::new_pentagonal_array(
                game.clone(),
                gfx::Transform {
                    translation: glm::Vec3::new(-127.0, 218.0, 0.0),
                    scale: glm::Vec3::new(18.0, -18.0, 18.0),
                    ..Default::default()
                },
                512
            ) as Arc<dyn gfx::Recordable>])
            .collect();

        util::run_async(move || {
            iproduct!(0..=64, 0..=64, 0..=64)
                .filter(|_| rng.gen_bool(0.0002))
                .for_each(|(x, y, z)| {
                    c_dm.insert_many_voxel([voxel2::WorldPosition(glm::I32Vec3::new(x, y, z))])
                });
        })
        .detach();

        util::run_async(move || {
            let it = iproduct!(-64..0, 0..64, 0..64)
                .map(|(x, y, z)| WorldPosition(glm::I32Vec3::new(x, y, z)));

            c_dm2.insert_many_voxel(it);

            load_model_from_file_into(
                glm::I32Vec3::new(0, 126, 0),
                &c_dm2,
                &dot_vox::load_bytes(include_bytes!("../../../models/menger.vox")).unwrap()
            );

            arbitrary_landscape_demo(&c_dm2);
        })
        .detach();

        let this = Arc::new(DemoScene {
            _dm:               dm.clone(),
            id:                util::Uuid::new(),
            _draws:            draws,
            lit_textured_cube: LitTextured::new_cube(
                game.clone(),
                gfx::Transform {
                    scale: glm::Vec3::new(5.0, 5.0, 5.0),
                    ..Default::default()
                }
            )
        });

        game.register(this.clone());

        this
    }
}

impl game::EntityCastDepot for DemoScene
{
    fn as_entity(&self) -> Option<&dyn game::Entity>
    {
        Some(self)
    }

    fn as_positionable(&self) -> Option<&dyn game::Positionable>
    {
        None
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        None
    }
}

impl game::Entity for DemoScene
{
    fn get_name(&self) -> Cow<'_, str>
    {
        "Test Scene".into()
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.id
    }

    fn tick(&self, _: &game::Game, _: game::TickTag)
    {
        self.lit_textured_cube.transform.lock().unwrap().translation = glm::Vec3::new(
            -40.0,
            unsafe { DEMO_FLOAT_HEIGHT.load(std::sync::atomic::Ordering::Relaxed) } + 64.0,
            30.0
        );
    }
}

fn load_model_from_file_into(world_offset: glm::I32Vec3, dm: &ChunkManager, data: &DotVoxData)
{
    let it = data.models[0]
        .voxels
        .iter()
        .map(|pos| WorldPosition(glm::U8Vec3::new(pos.x, pos.y, pos.z).cast() + world_offset));

    dm.insert_many_voxel(it);
}

fn arbitrary_landscape_demo(dm: &ChunkManager)
{
    let noise = noise::OpenSimplex::new(2384247834);

    let it = spiral::ChebyshevIterator::new(0, 0, 1024).map(|(x, z)| {
        WorldPosition(glm::I32Vec3::new(
            x,
            (noise.get([x as f64 / 256.0, z as f64 / 256.0]) * 256.0) as i32,
            z
        ))
    });

    // in a spiral formation stating at the center broding out, sample a height map
    // and collect a vector of the samplied points and a random color value

    dm.insert_many_voxel(it);
}
