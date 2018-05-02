#include "b2f_render.hpp"


int main(int argc, char* argv[]) {

    /**                         y
     * Left-Right ::= -X..+X    ^  z
     * Down-Up ::= -y..+y       | /
     * Near-Far ::= -z..+z      |/___> x
     */

    Scene mainScene{};

    SceneObject head{"head1"};
    head.add_component<Transform>(gm::Vec3f{0,0,0}, gm::Vec3f{0.2f,-0.2f,0}, gm::Vec3f{0.7,0.7,0.7});
    head.add_component<MeshRenderer>(argv[1]);

    SceneObject light{"light1"};
    light.add_component<Transform>(gm::Vec3f{0, 4, 0}, gm::Vec3f{0,0,0}, gm::Vec3f{1,1,1});
    light.add_component<DirectionLight>(gm::Vec3f{0,1,0.5});

    SceneObject camera{"cam1"};
    camera.add_component<Transform>(gm::Vec3f{0, 0, 1}, gm::Vec3f{0,0,0}, gm::Vec3f{1,1,1});
    camera.add_component<Camera>(gm::Vec3f{0,1,0}, gm::Vec3f{0, 0, 0});

    mainScene += head;
    mainScene += light;
    mainScene += camera;

    auto render = B2FRender{2000, 2000};

    render.render_scene(mainScene);
    render.save_zbuffer("zbuffer.tga");
    render.save_render("render.tga");

	return 0;
}