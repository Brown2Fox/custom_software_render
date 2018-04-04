
#include <iostream>
#include <limits>
#include <memory>

#include "mesh.hpp"
#include "geometry.hpp"
#include "tgaimage.hpp"
#include "b2f_features.hpp"


struct Pixel
{
	int x;
	int y;
	int r;
	int g;
	int b;

	Pixel()
	{
		x = 0;
		y = 0;
		r = 0;
		g = 0;
		b = 0;
	}

	Pixel(int x_, int y_, int r_, int g_, int b_)
	{
		x = x_;
		y = y_;
		r = r_;
		g = g_;
		b = b_;
	}
};



enum class ComponentType: int {
    Transform,
    Camera,
    Light,
    MeshRenderer,
    Nothing
};


class Component {

public:
    Component(): Type(ComponentType::Nothing) {}
    explicit Component(ComponentType type_) : Type(type_) {}
    const ComponentType Type;

};


class Transform: public Component {

    gm::Vec3f pos{0, 0, 0};
    gm::Vec3f rot{0, 0, 0};
    gm::Vec3f scale{0, 0, 0};

public:

    Transform() = default;

    Transform& operator=(const Transform& tr) {
        pos = tr.pos;
        rot = tr.rot;
        scale = tr.scale;
        return *this;
    }


    Transform(gm::Vec3f pos_, gm::Vec3f rot_, gm::Vec3f scale_) : Component(ComponentType::Transform)
    {
        pos = pos_;
        rot = rot_;
        scale = scale_;
    }


    gm::Mat4x4 getModelMatrix() {
        gm::Mat4x4 T = {{}};
        gm::Mat4x4 Rx = {{}};
        gm::Mat4x4 Ry = {{}};
        gm::Mat4x4 Rz = {{}};
        gm::Mat4x4 S = {{},{},{}};

        return T * (Rz * Ry * Rx) * S;
    }

};

class MeshRenderer: public Component
{

public:

    explicit MeshRenderer(std::string path) : Component(ComponentType::MeshRenderer)
    {
        mesh = std::make_shared<mesh::Mesh>(path);
    }

    std::shared_ptr<mesh::Mesh> mesh;


};


class DirectionLight: public Component {

    gm::Vec3f _direction{0, 0, 0};
public:

    DirectionLight() : Component(ComponentType::Light)
    {

    }

    explicit DirectionLight(const gm::Vec3f& direction_) : _direction(direction_), Component(ComponentType::Light)
    {

    }

    gm::Vec3f& getDirection() {
        return _direction;
    }

};

class Camera: public Component
{
    float r = 2.f;
    float l = -2.f;
    float t = 2.f;
    float b = -2.f;
    float f = 5.f;
    float n = -2.f;

    bool perspective = false;

public:

    Camera(): Component(ComponentType::Camera)
    {

    }

    Camera& operator=(const Camera& cam) {
        r = cam.r;
        l = cam.l;
        t = cam.t;
        b = cam.b;
        f = cam.f;
        n = cam.n;
        perspective = cam.perspective;
        return *this;
    }

    inline gm::Mat4x4 getViewMatrix()
    {
        return gm::Mat4x4::Identity();
    }

    inline gm::Mat4x4 persp()
    {
          return gm::Mat4x4{
                {2.f * n / (r - l), 0.f, (r + l) / (r - l), 0.f},
                {0.f, 2.f * n / (t - b), (t + b) / (t - b), 0.f},
                {0.f, 0.f, -(f + n)/(f - n), -2 * f/ (f - n)},
                {0.f, 0.f, -1, 0}
          };

    }

    inline gm::Mat4x4 ortho()
    {
        return gm::Mat4x4{
                {2.f / (r - l), 0.f, 0, -(r + l) / (r - l)},
                {0.f, 2.f / (t - b), 0, -(t + b) / (t - b)},
                {0.f, 0.f, 2.f / (f - n), -(f + n) / (f - n)},
                {0.f, 0.f, 0.f, 1}
        };
    }

    inline gm::Mat4x4 getProjectionMatrix() {
        return (perspective) ? persp() : ortho();
    }



};




class SceneObject {

    using TComponents = std::vector<std::shared_ptr<Component>>;

    Transform transform;
    TComponents components;

public:
    template<typename T, typename ...Targs>
    void addComponent(Targs... args) {
        components.push_back(shared new T(args...));
    }

    void setTransform(Transform& tr) {
        transform = tr;
    }

    Transform& getTransform() {
        return transform;
    }

    TComponents& getComponents() {
        return components;
    }
};


class Scene
{
    std::vector<SceneObject> static_objects;

public:
    void operator += (SceneObject& wo)
    {
        static_objects.push_back(wo);
    }

    std::vector<SceneObject>& getSceneObjects() {
        return static_objects;
    }
};



class B2FRender
{
    using RawImage = std::vector<Pixel>;
    using ZBuffer = std::vector<float>;

    gm::Vec3f light_dir{};

    gm::Mat4x4 PV{};
    size_t width{};
    size_t height{};

    ZBuffer z_buffer;
    RawImage img_buffer;
public:

    B2FRender(size_t h, size_t w) {
        width = w;
        height = h;
        z_buffer.resize(width*height, std::numeric_limits<float>::min());
        img_buffer.resize(width*height);
    }

    void rasterizeFace(std::vector<gm::Vec3f>& v, std::vector<gm::Vec2f>& vt, gm::Vec3f& fn) {
        float intensity = gm::dot(fn, light_dir);

        printf("%s %f, %f, %f\n", "fn", fn.x, fn.y, fn.z);
//        std::cout << "fn: " << fn << std::endl;

        if (v[0].y == v[1].y && v[0].y == v[2].y)
        {
            return;
        }
        if (v[0].y > v[1].y)
        {
            auto tmp = v[0];
            v[0] = v[1];
            v[1] = tmp;
            //std::swap(vt[0], vt[1]);
        }
        if (v[0].y > v[2].y)
        {
            auto tmp = v[0];
            v[0] = v[2];
            v[2] = tmp;
            //std::swap(vt[0], vt[2]);
        }
        if (v[1].y > v[2].y)
        {
            auto tmp = v[1];
            v[1] = v[2];
            v[2] = tmp;
            //std::swap(vt[1], vt[2]);
        }

        int A_height = int(v[2].y) - int(v[0].y);
        gm::Vec3f dir_A = v[2] - v[0];
        gm::Vec2f dir_vtA = vt[2] - vt[0];

        int B_downSegm_height = int(v[1].y) - int(v[0].y);
        gm::Vec3f dir_B_downSegm = v[1] - v[0];
        //gm::vec2 dir_vtB_first = vt[1] - vt[0];

        int B_upSegm_height = int(v[2].y) - int(v[1].y);
        gm::Vec3f dir_B_upSegm = v[2] - v[1];
        //gm::vec2 dir_vtB_second = vt[2] - vt[1];

        gm::Vec3f A{};
        gm::Vec3f B{};
        gm::Vec3f P{};
        //gm::vec2 vtB;

        auto start_y = int(v[0].y);
        auto end_y = int(v[2].y);

        //std::cout << A_height << " vs " << v[2].y - v[0].y << std::endl;


        for (int y = 0; y <= A_height; y++)
        {
            bool is_upSegm = y > B_downSegm_height || v[1].y == v[0].y;

            float segment_height = is_upSegm ? float(B_upSegm_height) : float(B_downSegm_height);

            float alpha =  float(y) / float(A_height);
            A = v[0] + (dir_A * alpha);
            //gm::vec2 vtA = vt[0] + dir_vtA * alpha;

            if (is_upSegm)
            {
                float beta = float(y - B_downSegm_height) / segment_height;
                B = v[1] + (dir_B_upSegm * beta);
                //vtB = vt[1] + dir_vtB_second * beta;
            }
            else // downSegm
            {
                float beta = float(y) / segment_height;
                B = v[0] + (dir_B_downSegm * beta);
                //vtB = vt[0] + dir_vtB_first * beta;
            }


            if (A.x > B.x)
            {
                std::swap(A, B);
                //std::swap(vtA, vtB);
            }

            auto start_x = int(A.x);
            auto end_x = int(B.x);
            int Py = start_y + y;

            for (int Px = start_x; Px <= end_x; Px++)
            {

                float phi = (end_x == start_x) ? 1.f : float(Px - start_x) / float(end_x - start_x);

                auto dir = B - A;
                P = A + (dir * phi);

                //gm::vec2 uvP = vtA + (vtB - vtA) * phi;

                int idx = Px + Py * width;

                //std::cout << "idx img: " << idx << "\n";

                if (idx >= 0 && idx < width*height)
                    if (P.z > z_buffer[idx])
                    {
                        z_buffer[idx] = P.z;
                        img_buffer[idx] = Pixel(Px, Py, int(intensity * 255), 0, 0 );
                    }

            }

        }

    }

    void renderScene(Scene& scene) {

        for (auto& so: scene.getSceneObjects()) {
            for(auto& component: so.getComponents()) {

                    switch (component->Type)
                    {
                        case ComponentType::Transform:
                            break;
                        case ComponentType::Light:
                        {
                            auto light = std::static_pointer_cast<DirectionLight>(component);
                            light_dir = light->getDirection();
                            break;
                        }


                        case ComponentType::Camera:
                        {
                            auto cam = std::static_pointer_cast<Camera>(component);
                            PV = cam->getProjectionMatrix() * cam->getViewMatrix();

                            break;
                        }
                        case ComponentType::MeshRenderer:
                        {
                            auto mr = std::static_pointer_cast<MeshRenderer>(component);

                            gm::Mat4x4 M = so.getTransform().getModelMatrix();

                            auto& faces = mr->mesh->faces;
                            auto& vertices = mr->mesh->vertices;
                            auto& txcoords = mr->mesh->txcoords;

                            gm::Mat4x4 MVP = PV * M;

                            // transform model vertices to render space
                            std::vector<mesh::Point> render_space{};
                            for (auto vert: vertices)
                            {
                                // wrap vertex into homogeneous coords
                                gm::Vec4f v = {vert.x, vert.y, vert.z, 1.f};
                                // projecting
                                gm::Vec4f v_p = MVP * v;
                                // convert homogeneous coords to normalized device coords (NDC)
                                //gm::vec3 v_ndc = gm::vec3(v_p.x / v_p.w, v_p.y / v_p.w, v_p.z / v_p.w);
                                gm::Vec3f v_ndc = {v_p.x / v_p.w, v_p.y / v_p.w, v_p.z / v_p.w};
                                // convert normalized device coords to screen space coords
                                render_space.push_back({(v_ndc.x) * width, (v_ndc.y + 0.5f) * height, v_ndc.z});
                            }

                            // render faces (triangles)
                            for (auto face: faces)
                            {
                                // calculate face normal
                                gm::Vec3f face_norm = {0, 0, 0};
                                for (auto norm_num : face.normals_nums)
                                {
                                    face_norm = face_norm + mr->mesh->normals[norm_num];
                                }
                                face_norm = gm::normalize(face_norm);


                                std::vector<gm::Vec3f> face_vertx;
                                for (auto vert_num: face.vertices_nums)
                                {
                                    face_vertx.push_back(render_space[vert_num]);
                                }

                                std::vector<gm::Vec2f> face_txcoord;
                                for (auto txcoord_num: face.txcoords_nums)
                                {
                                    face_txcoord.push_back(txcoords[txcoord_num]);
                                }

                                rasterizeFace(face_vertx, face_txcoord, face_norm);
                            }

                            break;
                        }

                        default:
                            break;
                    }
            }
        }
    }


    void saveToTga() {

        TGAImage image(width, height, TGAImage::GRAYSCALE);
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                image.set(i, j, TGAColor(z_buffer[i + j * width], 1));
            }
        }
        image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
        image.write_tga_file("zbuffer.tga");
    }

};


int main() {

//    Scene mainScene = Scene();
//
//    SceneObject head;
//
//    head.addComponent<Transform>(gm::Vec3f{0,0,0}, gm::Vec3f{0,0,0}, gm::Vec3f{1,1,1});
//    head.addComponent<MeshRenderer>(R"(D:\Workflow\B2FRender_kursach\assets\african_head.obj)");
//
//    SceneObject light;
//
//    light.addComponent<Transform>(gm::Vec3f{0, 0, 0}, gm::Vec3f{0,0,0}, gm::Vec3f{1,1,1});
//    light.addComponent<DirectionLight>(gm::Vec3f{0,0,-1});
//
//    SceneObject camera;
//
//    camera.addComponent<Transform>(gm::Vec3f{0.1, 0, 1}, gm::Vec3f{0,0,0}, gm::Vec3f{1,1,1});
//    camera.addComponent<Camera>();
//
//    mainScene += head;
//    mainScene += light;
//    mainScene += camera;
//
//    auto render = B2FRender(400, 400);

    gm::Vec3f a = {0, 12, 3};
    gm::Vec3f b = {0, 12, 3};

    gm::Vec3f c = a+b;

    std::cout << "C: " << c << std::endl;

    printf("%f, %f, %f\n", c.x, c.y, c.z);

    printf("%p %p", &c.x, &c[0]);

	return 0;
}