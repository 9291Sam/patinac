use std::sync::Arc;

use super::Entity;
use crate::game;

#[derive(Debug)]
pub struct TestScene
{
    objs: Vec<Arc<dyn gfx::Recordable>>,
    cube: Arc<gfx::lit_textured::LitTextured>,
    id:   util::Uuid
}

impl TestScene
{
    pub fn new(game: &game::Game) -> Arc<Self>
    {
        let mut objs: Vec<Arc<dyn gfx::Recordable>> = Vec::new();
        let mut cube: Option<Arc<gfx::lit_textured::LitTextured>> = None;

        for x in -5..=5
        {
            for z in -5..=5
            {
                objs.push(gfx::flat_textured::FlatTextured::new(
                    game.get_renderer(),
                    gfx::Vec3::new(x as f32, 0.0, z as f32),
                    gfx::flat_textured::FlatTextured::PENTAGON_VERTICES,
                    gfx::flat_textured::FlatTextured::PENTAGON_INDICES
                ));

                let a = gfx::lit_textured::LitTextured::new_cube(
                    game.get_renderer(),
                    gfx::Transform {
                        translation: gfx::Vec3::new(x as f32, 4.0, z as f32),
                        rotation:    *gfx::UnitQuaternion::from_axis_angle(
                            &gfx::Transform::global_up_vector(),
                            (x + z) as f32 / 4.0
                        ),
                        scale:       gfx::Vec3::repeat(0.4)
                    }
                );

                if x == 0 && z == 0
                {
                    cube = Some(a.clone());
                }

                objs.push(a);
            }
        }

        let this = Arc::new(TestScene {
            objs,
            cube: cube.unwrap(),
            id: util::Uuid::new()
        });

        game.register(this.clone());

        this
    }
}

impl Entity for TestScene
{
    fn get_name(&self) -> &str
    {
        "Test Scene"
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.id
    }

    fn get_position(&self) -> Option<gfx::Vec3>
    {
        None
    }

    fn tick(&self, game: &game::Game, _: crate::game::TickTag)
    {
        let mut guard = self.cube.transform.lock().unwrap();

        let quat = guard.rotation
            * *gfx::UnitQuaternion::from_axis_angle(
                &gfx::Transform::global_up_vector(),
                1.0 * game.get_delta_time()
            );

        guard.rotation = quat.normalize();
    }
}
