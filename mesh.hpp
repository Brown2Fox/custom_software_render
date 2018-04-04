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
        std::vector<int> vertices_nums;
        std::vector<int> normals_nums;
        std::vector<int> txcoords_nums;

        void set(unsigned int vertex_num, unsigned int normal_num, unsigned int txcoord_num, unsigned short ii)
        {
            vertices_nums[ii] = vertex_num-1;
            normals_nums[ii] = normal_num-1;
            txcoords_nums[ii] = txcoord_num-1;
        }

        void push_back(unsigned int vertex_num, unsigned int normal_num, unsigned int txcoord_num)
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
        std::vector<Vector> normals; // vn 1.0 1.0 1.0
        std::vector<Coord2d> txcoords; // vt 1.0 1.0

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

    };


    Mesh::Mesh(std::string filename) : vertices(), faces(), normals(), txcoords()
    {
        std::ifstream input;

        input.open(filename, std::ifstream::in);
        assert(input && "something wrong with input file");

        for (std::string line; std::getline(input, line);)
        {
            std::istringstream iss(line);
            char trash;
            if (line.compare(0, 2, "v ") == 0)
            {
                iss >> trash;
//                gm::Vec3f v{0, 0, 0};
                gm::Vec3f v{0, 0, 0};
                for (int i = 0; i < 3; i++) iss >> v[i];
                vertices.push_back(v);
                continue;
            }

            if (line.compare(0, 3, "vn ") == 0)
            {
                iss >> trash >> trash;
                gm::Vec3f vn = {0, 0, 0};
                for (size_t i = 0; i < 3; i++) iss >> vn[i];
                normals.push_back(vn);
                continue;
            }

            if (line.compare(0, 3, "vt ") == 0)
            {
                iss >> trash >> trash;
                gm::Vec2f vt = {0, 0};
                for (int i = 0; i < 2; i++) iss >> vt[i];
                txcoords.push_back(vt);
                continue;
            }

            //  f f1/ft1/fn1 f2/ft2/fn2 f3/ft3/fn3 f4/ft4/fn4 ...
            if (line.compare(0, 2, "f ") == 0)
            {

                Face f;
                int v = 0;
                int vt = 0;
                int vn = 0;
                iss >> trash;
                while (iss >> v >> trash >> vt >> trash >> vn)
                {
                    f.push_back(v, vt, vn);
                }

                faces.push_back(f);

                continue;
            }
        }

        std::cout << "# v# " << vertices.size() << " f# " << faces.size() << " vt# " << txcoords.size() << " vn# " << normals.size() << std::endl;
    }


    Mesh::~Mesh()
    {
    }

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
