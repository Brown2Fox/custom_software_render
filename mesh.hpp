#ifndef __MESH_H__
#define __MESH_H__

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "geometry.hpp"


namespace II
{
    enum
    {
        A = 0,
        B = 1,
        C = 2,
        D = 3
    };

    enum
    {
        X = 0,
        Y = 1,
        Z = 2,
        W = 3
    };
}

namespace mesh
{


    struct Face
    {
        std::vector<size_t> vertices_nums;
        std::vector<size_t> txcoords_nums;
        std::vector<size_t> normals_nums;

        void set(size_t vertex_num, size_t txcoord_num, size_t normal_num, unsigned short ii)
        {
            vertices_nums[ii] = vertex_num-1;
            normals_nums[ii] = normal_num-1;
            txcoords_nums[ii] = txcoord_num-1;
        }

        void push_back(size_t vertex_num, size_t txcoord_num, size_t normal_num)
        {
            vertices_nums.push_back(vertex_num - 1);
            normals_nums.push_back(normal_num - 1);
            txcoords_nums.push_back(txcoord_num - 1);
        }
    };


    using Point = gm::Vec3f;
    using Vector = gm::Vec3f;
    using Coord2d = gm::Vec2f;

    class Mesh
    {

    public:

        std::vector<Point> vertices;    // v 1.0 1.0 1.0
        std::vector<Coord2d> txcoords; // vt 1.0 1.0
        std::vector<Vector> normals; // vn 1.0 1.0 1.0

        //std::vector<Face> f_list;    //  f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 v4/vt4/vn4 ...

        std::vector<Face> faces;

        Mesh(std::string filename);

        ~Mesh();

        Point getV(int f_num, int v_num) const;

        Coord2d getVt(int ft_num, int vt_num) const;

        Vector getVn(int f_num, int vn_idx) const;

        int face_count() const;

        int vertex_count() const;

        inline Face get_face(int f_idx) const;

        std::tuple<size_t, size_t, size_t> getIndices(const std::string &line);

        bool isVertex(const std::string &line);

        bool isNormal(const std::string &line);

        bool isTexCoord(const std::string &line);

        bool isFace(const std::string &line);

    };


    Mesh::Mesh(std::string filename) : vertices(), faces(), normals(), txcoords()
    {
        std::ifstream input;

        input.open(filename, std::ifstream::in);
        assert(input && "something wrong with input file");

        for (std::string line; std::getline(input, line);)
        {
            std::istringstream iss(line);
            char trash[3];
            if (isVertex(line))
            {
                iss.read(trash, 2);
//                gm::Vec3f v{0, 0, 0};
                gm::Vec3f v{0, 0, 0};
                for (int i = 0; i < 3; i++) iss >> v[i];
                vertices.push_back(v);
                continue;
            }

            if (isNormal(line))
            {
                iss.read(trash, 3);
                gm::Vec3f vn = {0, 0, 0};
                for (size_t i = 0; i < 3; i++) iss >> vn[i];
                normals.push_back(vn);
                continue;
            }

            if (isTexCoord(line))
            {
                iss.read(trash, 3);
                gm::Vec2f vt = {0, 0};
                for (int i = 0; i < 2; i++) iss >> vt[i];
                txcoords.push_back(vt);
                continue;
            }

            //  f f1/ft1/fn1 f2/ft2/fn2 f3/ft3/fn3 f4/ft4/fn4 ...
            if (isFace(line))
            {
                Face f;

                iss.read(trash, 2);
                for (std::string str; std::getline(iss, str, ' ') || std::getline(iss, str);)
                {
                    auto indices = getIndices(str);
                    std::cout << std::get<0>(indices) << "/" << std::get<1>(indices) << "/" << std::get<2>(indices) << std::endl;
                    f.push_back(std::get<0>(indices), std::get<1>(indices), std::get<2>(indices));
                }

                faces.push_back(f);

                continue;
            }
        }

        std::cout << "# v# " << vertices.size() << " f# " << faces.size() << " vt# " << txcoords.size() << " vn# " << normals.size() << std::endl;
    }


    std::tuple<size_t, size_t, size_t> Mesh::getIndices(const std::string &line)
    {
        size_t v = 0;
        size_t vt = 0;
        size_t vn = 0;
        std::string value;
        std::stringstream source(line);

        std::getline(source, value, '/');
        v = value.empty() ? 0 : std::stoul(value);
        std::getline(source, value, '/');
        vt = value.empty() ? 0 : std::stoul(value);
        std::getline(source, value, ' ');
        vn = value.empty() ? 0 : std::stoul(value);

        return std::make_tuple(v, vt, vn);
    }

    bool Mesh::isVertex(const std::string &line)
    {
        return line[0] == 'v' && line[1] == ' ';
    }

    bool Mesh::isNormal(const std::string &line)
    {
        return line[0] == 'v' && line[1] == 'n';
    }

    bool Mesh::isTexCoord(const std::string &line)
    {
        return line[0] == 'v' && line[1] == 't';
    }

    bool Mesh::isFace(const std::string &line)
    {
        return line[0] == 'f';
    }


    Mesh::~Mesh() = default;

    gm::Vec3f Mesh::getV(int f_num, int v_num) const
    {
        return vertices[faces[f_num].vertices_nums[v_num]];
    }

    gm::Vec2f Mesh::getVt(int f_num, int vt_num) const
    {
        return txcoords[faces[f_num].txcoords_nums[vt_num]];
    }

    gm::Vec3f Mesh::getVn(int f_num, int vn_num) const
    {
        return normals[faces[f_num].normals_nums[vn_num]];
    }

    int Mesh::face_count() const
    {
        return faces.size();
    }


    int Mesh::vertex_count() const
    {
        return vertices.size();
    }


    inline Face Mesh::get_face(int idx) const
    {
        return faces[idx];
    }

}

#endif //__MESH_H__
