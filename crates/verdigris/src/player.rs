use rapier3d::prelude::RigidBody;

struct Player {}

impl game::Entity for Player
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        todo!()
    }

    fn get_uuid(&self) -> util::Uuid
    {
        todo!()
    }

    fn tick(&self, game: &game::Game, _: game::TickTag)
    {
        todo!()
    }
}

impl game::EntityCastDepot for Player
{
    fn as_self_managed(
        self: std::sync::Arc<Self>
    ) -> Option<std::sync::Arc<dyn game::SelfManagedEntity>>
    {
        todo!()
    }

    fn as_positionalable(&self) -> Option<&dyn game::Positionalable>
    {
        todo!()
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        todo!()
    }

    fn as_collideable(&self) -> Option<&dyn game::Collideable>
    {
        todo!()
    }
}

impl game::Positionalable for Player
{
    fn get_position(&self) -> gfx::glm::Vec3
    {
        todo!()
    }
}

impl game::Transformable for Player
{
    fn get_transform(&self) -> gfx::Transform
    {
        todo!()
    }
}

impl game::Collideable for Player
{
    fn init_collideable(&self) -> RigidBody
    {
        todo!()
    }

    fn physics_tick(&self, rigid_body: &mut RigidBody, game: &game::Game, _: game::TickTag)
    {
        todo!()
    }
}
