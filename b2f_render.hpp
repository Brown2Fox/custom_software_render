#ifndef B2FRENDER_B2F_RENDER_HPP
#define B2FRENDER_B2F_RENDER_HPP


#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <map>

#include "mesh.hpp"
#include "geometry.hpp"
#include "tgaimage.hpp"
#include "b2f_features.hpp"

template<typename TInt>
struct RGBAColor
{
    TInt r{0};
    TInt g{0};
    TInt b{0};
    TInt a{255};

    RGBAColor() = default;
    RGBAColor(TInt _r, TInt _g, TInt _b, TInt _a = 255): r{_r}, g{_g}, b{_b}, a{_a} {}
};

template<typename TInt>
std::ostream& operator << (std::ostream &out, RGBAColor<TInt>& clr)
{
    out << "RGBA:{" << clr.r << "," << clr.g << "," << clr.b << "," << clr.a << "}";
    return out;
}


template<typename TInt, typename TColor>
struct Pixel
{
    TInt x{0};
    TInt y{0};
    TColor color{};

    Pixel() = default;
    Pixel(TInt _x, TInt _y, TColor _color): x{_x}, y{_y}, color{_color} {}
};

class SceneObject;

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

    explicit Component(ComponentType _type) :
            Type{_type}
    {}

    void set_parent(SceneObject* _parent)
    {
        parent = _parent;
    }

    SceneObject* get_parent()
    {
        return parent;
    }

    const ComponentType Type;

protected:
    SceneObject* parent;
};


class Transform: public Component {

    gm::Vec3f pos_{0, 0, 0};
    gm::Vec3f rot_{0, 0, 0};
    gm::Vec3f scale_{0, 0, 0};

public:

    Transform() = default;

    Transform& operator=(const Transform& _tr) {
        pos_ = _tr.pos_;
        rot_ = _tr.rot_;
        scale_ = _tr.scale_;
        return *this;
    }


    Transform(const gm::Vec3f& _pos, const gm::Vec3f& _rot, const gm::Vec3f& _scale) : Component(ComponentType::Transform)
    {
        pos_ = _pos;
        rot_ = _rot;
        scale_ = _scale;
    }


    inline gm::Mat4x4 get_translation_mat() {
        return {{1, 0, 0, pos_.x},
                {0, 1, 0, pos_.y},
                {0, 0, 1,  pos_.z},
                {0, 0, 0, 1}};
    }

    inline gm::Mat4x4 get_rotation_mat() {
        using std::sin;
        using std::cos;

        gm::Mat4x4 Rx = {{1, 0, 0, 0},
                         {0, cos(rot_.x), sin(-rot_.x), 0},
                         {0, sin(rot_.x), cos(rot_.x), 0},
                         {0, 0, 0, 1}};
        gm::Mat4x4 Ry = {{cos(rot_.y), 0, sin(rot_.y), 0},
                         {0, 1, 0, 0},
                         {sin(rot_.y), 0, cos(rot_.y), 0},
                         {0, 0, 0, 1}};
        gm::Mat4x4 Rz = {{cos(rot_.z), sin(-rot_.z), 0, 0},
                         {sin(rot_.z), cos(rot_.z), 0, 0},
                         {0, 0, 1, 0},
                         {0, 0, 0, 1}};

        return Rz * Ry * Rx;
    }

    inline gm::Mat4x4 get_scale_mat() {
        return {{scale_.x, 0, 0, 0},
                {0, scale_.y, 0, 0},
                {0, 0, scale_.z, 0},
                {0, 0, 0, 1}};
    }

    gm::Mat4x4 get_model_mat() {

        gm::Mat4x4 T = get_translation_mat();
        gm::Mat4x4 R = get_rotation_mat();
        gm::Mat4x4 S = get_scale_mat();

        return T * R * S;
    }

    gm::Vec3f get_pos()
    {
        return pos_;
    }

};

class MeshRenderer: public Component
{

public:

    explicit MeshRenderer(const std::string& _path) : Component(ComponentType::MeshRenderer)
    {
        mesh = std::make_shared<mesh::Mesh>(_path);
    }

    std::shared_ptr<mesh::Mesh> mesh;

};


class DirectionLight: public Component {

    gm::Vec3f direction_{0, 0, 0};
public:

    DirectionLight(): Component(ComponentType::Light) {}

    explicit DirectionLight(const gm::Vec3f& _direction):  Component(ComponentType::Light)
    {
        direction_ = _direction;
    }

    gm::Vec3f get_direction() {
        return direction_;
    }

};



class SceneObject {

    using TComponents = std::vector<std::shared_ptr<Component>>;

    std::shared_ptr<Transform> transform_;
    TComponents components_;
    std::string name_;


public:


    SceneObject(const std::string& _name): name_{_name} {}

    template<typename T, typename ...Targs>
    void add_component(Targs... args) {
        auto tmp = std::make_shared<T>(args...);
        tmp->set_parent(this);
        components_.push_back(tmp);
    }

    std::shared_ptr<Transform> transform() {
        for (auto &co: components_)
        {
            if (co->Type == ComponentType::Transform)
                return std::static_pointer_cast<Transform>(co);
        }
        return transform_;
    }

    TComponents& components() {
        return components_;
    }

    std::string get_name() const
    {
        return name_;
    }
};

class Camera: public Component
{
    float r_ = 2.f;
    float l_ = -2.f;
    float t_ = 2.f;
    float b_ = -2.f;
    float f_ = 5.f;
    float n_ = -2.f;

    bool perspective_ = false;

    gm::Vec3f up_{0,0,1};
    gm::Vec3f eye_{0,1,0};

public:

    Camera(): Component{ComponentType::Camera} {}

    Camera(const gm::Vec3f& _up, const gm::Vec3f& _eye):
            Component{ComponentType::Camera},
            up_{_up},
            eye_{_eye}
    {
    }


    Camera& operator= (const Camera& _cam) {
        r_ = _cam.r_;
        l_ = _cam.l_;
        t_ = _cam.t_;
        b_ = _cam.b_;
        f_ = _cam.f_;
        n_ = _cam.n_;
        perspective_ = _cam.perspective_;
        return *this;
    }

    inline gm::Mat4x4 get_view_mat()
    {
        auto pos = parent->transform()->get_pos();
        auto dir = eye_ - pos;
        auto right_ = gm::normalize(gm::cross(dir, up_));

//        std::cout << dir << std::endl;
//        std::cout << right_ << std::endl;

        return gm::Mat4x4{
                {right_.x,up_.x,eye_.x,0},
                {right_.y,up_.y,eye_.y,0},
                {right_.z,up_.z,eye_.z,0},
                {-(pos*right_),-(pos*up_),-(pos*eye_),1}
        };
    }


    inline gm::Mat4x4 get_persp_mat()
    {
        return gm::Mat4x4{
                {2.f * n_ / (r_ - l_), 0.f, (r_ + l_) / (r_ - l_), 0.f},
                {0.f, 2.f * n_ / (t_ - b_), (t_ + b_) / (t_ - b_), 0.f},
                {0.f, 0.f, -(f_ + n_)/(f_ - n_), -2 * f_/ (f_ - n_)},
                {0.f, 0.f, -1, 0}
        };

    }

    inline gm::Mat4x4 get_ortho_mat()
    {
        return gm::Mat4x4{
                {2.f / (r_ - l_), 0.f, 0, -(r_ + l_) / (r_ - l_)},
                {0.f, 2.f / (t_ - b_), 0, -(t_ + b_) / (t_ - b_)},
                {0.f, 0.f, 2.f / (f_ - n_), -(f_ + n_) / (f_ - n_)},
                {0.f, 0.f, 0.f, 1}
        };
    }

    inline gm::Mat4x4 get_projection_mat() {
        return (perspective_) ? get_persp_mat() : get_ortho_mat();
    }

};



class Scene
{
    std::vector<SceneObject> static_objects_;



public:
    void operator += (SceneObject& wo)
    {
        static_objects_.push_back(wo);

        std::hash<std::string> hasher{};
        size_t hash{hasher(wo.get_name())};

        for (auto& comp: wo.components())
        {
            switch (comp->Type)
            {
                case ComponentType::Transform:
                    transforms.insert(std::make_pair(hash, std::static_pointer_cast<Transform>(comp)));
                    break;

                case ComponentType::MeshRenderer:
                    mesh_renderers.insert(std::make_pair(hash, std::static_pointer_cast<MeshRenderer>(comp)));
                    break;

                case ComponentType::Camera:
                    cameras.insert(std::make_pair(hash, std::static_pointer_cast<Camera>(comp)));
                    break;

                case ComponentType::Light:
                    lights.insert(std::make_pair(hash, std::static_pointer_cast<DirectionLight>(comp)));
                    break;

                default:
                    break;
            }
        }
    }

    template<typename F, typename T>
    F find(const T& by_elem)
    {

    };


    std::vector<SceneObject>& scene_objects() {
        return static_objects_;
    }


    std::map<size_t, std::shared_ptr<Transform>> transforms;
    std::map<size_t, std::shared_ptr<Camera>> cameras;
    std::map<size_t, std::shared_ptr<MeshRenderer>> mesh_renderers;
    std::map<size_t, std::shared_ptr<DirectionLight>> lights;

};



class B2FRender
{
    using RGBAPixel = Pixel<int, RGBAColor<uint8_t>>;
    using RawImage = std::vector<RGBAPixel>;
    using ZBuffer = std::vector<float>;

    gm::Vec3f light_dir{};

    gm::Mat4x4 P{};
    gm::Mat4x4 V{};
    size_t width{};
    size_t height{};

    ZBuffer z_buffer;
    RawImage img_buffer;
    float min_z_buffer;
    float max_z_buffer;
    int calls = 0;

public:

    B2FRender(size_t h, size_t w) {
        width = w;
        height = h;
        z_buffer.resize(width*height, std::numeric_limits<float>::lowest());
        min_z_buffer = std::numeric_limits<float>::max();
        max_z_buffer = std::numeric_limits<float>::lowest();
        img_buffer.resize(width*height);
    }

    void rasterizeFace(std::vector<gm::Vec3f>& v, std::vector<gm::Vec2f>& vt, gm::Vec3f& fn) {

        calls++;

//        printf("calls - %i\n", calls);

        float intensity = gm::dot(fn, light_dir) * 70;

        if (v[0].y == v[1].y && v[0].y == v[2].y)
        {
            return;
        }
        if (v[0].y > v[1].y)
        {
            std::swap(v[0], v[1]);
        }
        if (v[0].y > v[2].y)
        {
            std::swap(v[0], v[2]);
        }
        if (v[1].y > v[2].y)
        {
            std::swap(v[1], v[2]);
        }
        gm::Vec3f& start_p = v[0];
        gm::Vec3f& middle_p = v[1];
        gm::Vec3f& end_p = v[2];

        int A_height = std::lrint(end_p.y) - std::lrint(start_p.y);
        float f_A_height = static_cast<float>(A_height);
        gm::Vec3f dir_A = end_p - start_p;
//        gm::Vec2f dir_vtA = vt[2] - vt[0];

        int B_downSegm_height = std::lrint(middle_p.y) - std::lrint(start_p.y);
        float f_B_downSegm_height = static_cast<float>(B_downSegm_height);
        gm::Vec3f dir_B_downSegm = middle_p - start_p;
        //gm::vec2 dir_vtB_first = vt[1] - vt[0];

        int B_upSegm_height = std::lrint(end_p.y) - std::lrint(middle_p.y);
        float f_B_upSegm_height = static_cast<float>(B_upSegm_height);
        gm::Vec3f dir_B_upSegm = end_p - middle_p;
        //gm::vec2 dir_vtB_second = vt[2] - vt[1];

        assert(A_height == B_downSegm_height + B_upSegm_height);

        //gm::vec2 vtB;

        int start_y = std::lrint(start_p.y);
        int end_y = std::lrint(end_p.y);

        for (int y = 0; y <= A_height; y++)
        {
            bool is_upSegm = (y > B_downSegm_height) || (middle_p.y == start_p.y);

            const gm::Vec3f start_A = start_p;
            const int y_A = y;

            const float f_B_height = is_upSegm ? f_B_upSegm_height : f_B_downSegm_height;
            const gm::Vec3f& dir_B = is_upSegm ? dir_B_upSegm : dir_B_downSegm;
            const gm::Vec3f& start_B = is_upSegm ? middle_p : start_p;
            const int y_B = is_upSegm ? y - B_downSegm_height : y;

            float alpha =  y_A / f_A_height;
            gm::Vec3f A = {start_A + (dir_A * alpha)};

            float beta = y_B / f_B_height;
            gm::Vec3f B = {start_B + (dir_B * beta)};
            //gm::vec2 vtA = vt[0] + dir_vtA * alpha;


            if (A.x > B.x)
            {
                std::swap(A, B);
                //std::swap(vtA, vtB);
            }

            int start_x = std::lrint(A.x);
            int end_x = std::lrint(B.x);
            int Py = start_y + y;

            for (int Px = start_x; Px <= end_x; Px++)
            {

//                printf("Py: %i, Px: %i, calls - %i\n", Py, Px, calls);
                float phi = (end_x == start_x) ? 1.f : (Px - start_x) / static_cast<float>(end_x - start_x);
                gm::Vec3f dir = B - A;
                gm::Vec3f P = {A + (dir * phi)}; // Point on rendering triangle

                //gm::vec2 uvP = vtA + (vtB - vtA) * phi;

                size_t idx = Px + Py * width;

                //std::cout << "idx img: " << idx << "\n";

                if (idx < width*height)
                {
                    if (P.z > z_buffer[idx])
                    {
                        if (P.z > max_z_buffer) max_z_buffer = P.z;
                        if (P.z < min_z_buffer) min_z_buffer = P.z;
                        z_buffer[idx] = P.z;

                        auto color = static_cast<uint8_t>(100*std::abs(intensity) + 50);
                        color = (color < 0) ? 0 : color;
                        color = (color > 255) ? 255 : color;

                        if (intensity > 0)
                        {
                            img_buffer[idx] = RGBAPixel{Px, Py, {color, 0, 0}};
                        }
                        else
                        {
                            img_buffer[idx] = RGBAPixel{Px, Py, {0, color, color}};
                        }

                    }
                }

            }

        }

    }

    void render_scene(Scene &scene) {

        for (auto& pair: scene.cameras)
        {
            auto cam = pair.second;
            P = cam->get_projection_mat();
            V = cam->get_view_mat();
            break;
        }

        for (auto& pair: scene.lights)
        {
            auto light = pair.second;
            light_dir = light->get_direction();
        }


        for(auto& pair: scene.mesh_renderers)
        {
            auto mr = pair.second;
            auto hash = pair.first;

            auto tr = scene.transforms.find(hash)->second;

            gm::Mat4x4 M = tr->get_model_mat();

            auto& faces = mr->mesh->faces;
            auto& vertices = mr->mesh->vertices;
            auto& txcoords = mr->mesh->txcoords;
            auto& normals = mr->mesh->normals;

            gm::Mat4x4 MVP = P * V * M;

            // transform model vertices to render space
            std::vector<mesh::Point> render_space{};
            for (auto vert: vertices)
            {
                // wrap vertex into homogeneous coords
                gm::Vec4f v = {vert.x, vert.y, vert.z, 1.f};
                // projecting
                //gm::Vec4f v_p = MVP * v;
                gm::Vec4f v_p = P * M * v;
                // convert homogeneous coords to normalized device coords (NDC)
                //gm::vec3 v_ndc = gm::vec3(v_p.x / v_p.w, v_p.y / v_p.w, v_p.z / v_p.w);
                gm::Vec3f v_ndc = {v_p.x / v_p.w, v_p.y / v_p.w, v_p.z / v_p.w};
                // convert normalized device coords to screen space coords
                render_space.push_back({(v_ndc.x + 0.5f) * width, (v_ndc.y + 0.5f) * height, v_ndc.z});
            }

            // render faces (triangles)
            for (auto face: faces)
            {
                // calculate face normal
//                auto face_side_1 = vertices[face.vertices_nums[1]] - vertices[face.vertices_nums[0]];
//                auto face_side_2 = vertices[face.vertices_nums[2]] - vertices[face.vertices_nums[0]];
//                gm::Vec3f face_norm = gm::normalize(gm::cross(face_side_1, face_side_2));

                gm::Vec3f face_norm{};
                for (auto norm_num: face.normals_nums)
                {
                    face_norm = face_norm + normals[norm_num];
                }
                face_norm = gm::normalize(face_norm);

                std::vector<gm::Vec3f> face_vertx;
                for (auto vert_num: face.vertices_nums)
                {
                    face_vertx.push_back(render_space[vert_num]);
                }

                std::vector<gm::Vec2f> face_txcoord;
//                for (auto txcoord_num: face.txcoords_nums)
//                {
//                    face_txcoord.push_back(txcoords[txcoord_num]);
//                }

                rasterizeFace(face_vertx, face_txcoord, face_norm);
            }

        }
    }


    void save_render(const std::string &out_file_name) {

        TGAImage image{width, height, TGAImage::RGB};

        for (auto& p: img_buffer)
        {
            image.set(p.x, p.y, TGAColor{p.color.r, p.color.g, p.color.b});
        }
        image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
        image.write_tga_file(out_file_name.c_str());
    }

    void save_zbuffer(const std::string &out_file_name)
    {
        TGAImage image{width, height, TGAImage::GRAYSCALE};
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                auto zb = z_buffer[i + j * width];
                if (zb != std::numeric_limits<float>::lowest())
                {
                    zb = (zb + std::abs(min_z_buffer)) * 255.0f/(max_z_buffer-min_z_buffer);
                    image.set(i, j, TGAColor(static_cast<int>(zb), 1));
                }
            }
        }
        image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
        image.write_tga_file(out_file_name.c_str());
    }

};




#endif //B2FRENDER_B2F_RENDER_HPP
