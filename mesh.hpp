#ifndef __MESH_H__
#define __MESH_H__

#include <vector>

struct Face
{
	std::vector<int> v_indicies;
	std::vector<int> vn_indicies;
	std::vector<int> vt_indicies;

	void set(uint32_t vlist_idx, uint32_t vnlist_idx, uint32_t vtlist_idx, uint32_t idx)
	{
		v_indicies[idx] = vlist_idx-1;
		vn_indicies[idx] = vnlist_idx-1;
		vt_indicies[idx] = vtlist_idx-1;
	}

	void push_back(uint32_t vlist_idx, uint32_t vnlist_idx, uint32_t vtlist_idx)
	{
		v_indicies.push_back(vlist_idx - 1);
		vn_indicies.push_back(vnlist_idx - 1);
		vt_indicies.push_back(vtlist_idx - 1);
	}
};

class Mesh
{

public:

	std::vector<glm::vec3> v_list;	// v 1.0 1.0 1.0
	std::vector<glm::vec3> vn_list; // vn 1.0 1.0 1.0
	std::vector<glm::vec2> vt_list; // vt 1.0 1.0
	std::vector<Face> f_list;	//  f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 v4/vt4/vn4 ...

    explicit Mesh(const char *filename);
	~Mesh();

	glm::vec3 getV(int face_idx, int v_idx) const;
	glm::vec2 getVt(int face_idx, int vt_idx) const;
	glm::vec3 getVn(int face_idx, int vn_idx) const;
	int nfaces() const;
	int nverts() const;
	inline Face get_face(int f_idx) const;

};




Mesh::Mesh(const char *filename) : v_list(), f_list(), vn_list(), vt_list()
{
    std::ifstream in;
    in.open (filename, std::ifstream::in);
    if (in.fail()) return;
    std::string line;
    while (!in.eof())
	{
        std::getline(in, line);
        std::istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v "))
		{
            iss >> trash;
            glm::vec3 v;
            for (int i = 0; i < 3; i++) iss >> v[i];
            v_list.push_back(v);
        }
		else if (!line.compare(0, 3, "vn "))
		{
            iss >> trash >> trash;
			glm::vec3 vn;
            for (int i = 0; i < 3; i++) iss >> vn[i];
            vn_list.push_back(vn);
        }
		else if (!line.compare(0, 3, "vt "))
		{
            iss >> trash >> trash;
            glm::vec2 vt;
            for (int i = 0; i < 2; i++) iss >> vt[i];
            vt_list.push_back(vt);
        }
		else if (!line.compare(0, 2, "f "))
		{
			Face f;
			int v = 0;
			int vt = 0;
			int vn = 0;
			iss >> trash;
            while (iss >> v >> trash >> vt >> trash >> vn)
			{
                f.push_back(v, vn, vt);
            }
            f_list.push_back(f);
        }
    }
    std::cout << "# v# " << v_list.size() << " f# "  << f_list.size() << " vt# " << vt_list.size() << " vn# " << vn_list.size() << std::endl;
}



Mesh::~Mesh() {
}

glm::vec3 Mesh::getV(int face_idx, int v_idx) const
{
	return v_list[f_list[face_idx].v_indicies[v_idx]];
}

glm::vec2 Mesh::getVt(int face_idx, int vt_idx) const
{
	return vt_list[f_list[face_idx].vt_indicies[vt_idx]];
}

glm::vec3 Mesh::getVn(int face_idx, int vn_idx) const
{
	return vn_list[f_list[face_idx].vn_indicies[vn_idx]];
}

int Mesh::nfaces() const
{
	return f_list.size();
}


int Mesh::nverts() const
{
	return v_list.size();
}


inline Face Mesh::get_face(int idx) const
{
	return f_list[idx];
}



#endif //__MESH_H__
