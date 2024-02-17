use std::sync::Arc;

use gfx::glm;

use super::Entity;

#[derive(Debug)]
pub struct TestScene
{
    _objs:       Vec<Arc<dyn gfx::Recordable>>,
    rotate_objs: Vec<Arc<gfx::LitTextured>>,
    id:          util::Uuid
}

impl TestScene
{
    pub fn new(game: &super::Game) -> Arc<Self>
    {
        let mut objs: Vec<Arc<dyn gfx::Recordable>> = Vec::new();
        let mut rotate_objs: Vec<Arc<gfx::LitTextured>> = Vec::new();

        for x in -5..=5
        {
            for z in -5..=5
            {
                objs.push(gfx::FlatTextured::new(
                    game.get_renderer(),
                    glm::Vec3::new(x as f32, 0.0, z as f32),
                    gfx::FlatTextured::PENTAGON_VERTICES,
                    gfx::FlatTextured::PENTAGON_INDICES
                ));

                let a = gfx::LitTextured::new_cube(
                    game.get_renderer(),
                    gfx::Transform {
                        translation: glm::Vec3::new(x as f32, 4.0, z as f32),
                        rotation:    *glm::UnitQuaternion::from_axis_angle(
                            &gfx::Transform::global_up_vector(),
                            (x + z) as f32 / 4.0
                        ),
                        scale:       glm::Vec3::repeat(0.4)
                    }
                );

                if x == 0 && z == 0
                {
                    rotate_objs.push(a.clone());
                }

                objs.push(a);
            }
        }

        let this = Arc::new(TestScene {
            _objs: objs,
            rotate_objs,
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

    fn get_position(&self) -> Option<glm::Vec3>
    {
        None
    }

    fn tick(&self, game: &super::Game, _: super::TickTag)
    {
        for o in &self.rotate_objs
        {
            let mut guard = o.transform.lock().unwrap();

            let quat = guard.rotation
                * *glm::UnitQuaternion::from_axis_angle(
                    &gfx::Transform::global_up_vector(),
                    1.0 * game.get_delta_time()
                );

            guard.rotation = quat.normalize();
        }
    }
}
