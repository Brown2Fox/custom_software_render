#include "b2f_render_jni.h"

#include <iostream>
#include <omp.h>
#include "Mesh.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp>


JNIEXPORT void JNICALL Java_b2f_lib_B2FLib_calc(JNIEnv *pEnv, jclass pClass, jint a)
{
	omp_set_num_threads(a);

	#pragma omp parallel
	{
		printf("Threads: %i\n", omp_get_num_threads());
	};
}

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

void rasterizeTriangle(glm::vec3 v[3], glm::vec2 vt[3], std::vector<Pixel> &image, float intensity, float *zbuffer, int width, int height)
{


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
	glm::vec3 dir_A = v[2] - v[0];
	glm::vec2 dir_vtA = vt[2] - vt[0];

	int B_downSegm_height = int(v[1].y) - int(v[0].y);
	glm::vec3 dir_B_downSegm = v[1] - v[0];
	//glm::vec2 dir_vtB_first = vt[1] - vt[0];

	int B_upSegm_height = int(v[2].y) - int(v[1].y);
	glm::vec3 dir_B_upSegm = v[2] - v[1];
	//glm::vec2 dir_vtB_second = vt[2] - vt[1];

	glm::vec3 A{};
	glm::vec3 B{};
	glm::vec3 P{};
	//glm::vec2 vtB;

	auto start_y = int(v[0].y);
	auto end_y = int(v[2].y);

	//std::cout << A_height << " vs " << v[2].y - v[0].y << std::endl;


	for (int y = 0; y <= A_height; y++)
	{
		bool is_upSegm = y > B_downSegm_height || v[1].y == v[0].y;

		float segment_height = is_upSegm ? float(B_upSegm_height) : float(B_downSegm_height);

		float alpha =  float(y) / float(A_height);
		A = v[0] + (dir_A * alpha);
		//glm::vec2 vtA = vt[0] + dir_vtA * alpha;

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

			//glm::vec2 uvP = vtA + (vtB - vtA) * phi;

			int idx = Px + Py * width;

			//std::cout << "idx img: " << idx << "\n";

			if (idx >= 0 && idx < width*height)
			if (P.z > zbuffer[idx])
			{
				zbuffer[idx] = P.z;
				image[idx] = Pixel(Px, Py, int(intensity * 255), 0, 0 );
			}

		}

	}

}


inline glm::mat4x4 persp(float r, float l, float b, float t, float n, float f)
{
	return glm::mat4x4(glm::vec4(2. * n / (r - l), 0., (r + l) / (r - l), 0.),
					   glm::vec4(0., 2. * n / (t - b), (t + b) / (t - b), 0.),
					   glm::vec4(0., 0., -(f + n)/(f - n), -2 * f/ (f - n)),
					   glm::vec4(0., 0., -1, 0));
}

inline glm::mat4x4 ortho(float l, float r, float b, float t, float n, float f)
{
	return glm::mat4x4(
		glm::vec4(2. / (r - l), 0., 0, -(r + l) / (r - l)),
		glm::vec4(0., 2. / (t - b), 0, -(t + b) / (t - b)),
		glm::vec4(0., 0., 2. / (f - n), -(f + n) / (f - n)),
		glm::vec4(0., 0., 0., 1)
	);
}


JNIEXPORT jobjectArray JNICALL Java_b2f_lib_B2FLib_render(JNIEnv *pEnv, jclass pClass, jint width, jint height, jint depth)
{
	std::FILE* logFile;
	fopen_s(&logFile, "log.txt", "w+");
	fprintf(logFile, "==========HALO!!!=======\n");

	jclass jPixel = pEnv->FindClass("b2f/Pixel");

	jint fullSize = width * height;
	jobjectArray jImage = pEnv->NewObjectArray(fullSize, jPixel, nullptr);



	jmethodID jPixel_init = pEnv->GetMethodID(jPixel, "<init>", "(IIIII)V");
	if (nullptr == jPixel_init) return nullptr;


	Mesh *pMesh = new Mesh("D:/Users/Antti/Workflow/BBB/obj/african_head.obj");

	float r = 2.f;
	float l = -2.f;
	float t = 2.f;
	float b = -2.f;
	float f = 5.f;
	float n = -2.f;

	glm::mat4x4 P = ortho(l, r, b, t, n, f);
	glm::mat4x4 M = glm::mat4x4(1.0);
	glm::mat4x4 V = glm::mat4x4(1.0);

	glm::mat4x4 MVP = P * V * M;

	std::vector<glm::vec3> screen_space_coords(pMesh->nverts());

	#pragma omp parallel for shared(pMesh, screen_space_coords)
	for (int i = 0; i < pMesh->nverts(); i++)
	{
		// wrap vertex into homogeneous coords
		glm::vec4 v = glm::vec4(pMesh->v_list[i], 1);
		// projecting
		glm::vec4 v_p = MVP * v;
		// convert homogeneous coords to normalized device coords (NDC)
		//glm::vec3 v_ndc = glm::vec3(v_p.x / v_p.w, v_p.y / v_p.w, v_p.z / v_p.w);
		glm::vec3 v_ndc = glm::vec3(v_p.x, v_p.y, v_p.z);
		// convert normalized device coords to screen space coords
		screen_space_coords[i] = glm::vec3((v_ndc.x)* width, (v_ndc.y+0.5) * height, v_ndc.z);
	}


    auto *zbuffer = new float[width * height];
	std::cout << "image size: " << width * height << "\n";


	std::vector<Pixel> image(static_cast<unsigned long long int>(width * height));

	for (int i = 0; i < width * height; i++)
	{
		zbuffer[i] = std::numeric_limits<float>::min();
	}

	auto light_dir = glm::normalize(glm::vec3(0.1, 0, 1));

	for (int i = 0; i < pMesh->nfaces(); i++)
	{
		glm::vec3 vn(0);

		//printf("face %i of %i\n", i, pMesh->nfaces());

		for (auto idx : pMesh->f_list[i].vn_indicies)
		{
			vn += pMesh->vn_list[idx];
		}

		vn = glm::normalize(vn);

		float intensity = glm::dot(vn, light_dir);

		if (intensity > 0)
		{

			glm::vec2 vt_triangle[3];
			glm::vec3 v_triangle[3];

			for (int k = 0; k < 3; k++)
			{
				vt_triangle[k] = pMesh->getVt(i, k);
				v_triangle[k] = screen_space_coords[pMesh->f_list[i].v_indicies[k]];
			}

			rasterizeTriangle(v_triangle, vt_triangle, image, intensity, zbuffer, width, height);
		}

	}

	delete[] zbuffer;

	for (int i = 0; i < fullSize; i++)
	{
		//printf("pix %i of %i rgb(%i,%i,%i)\n", i, fullSize, image[i].r, image[i].g, image[i].b);

		jobject p = pEnv->NewObject(jPixel, jPixel_init, image[i].x, image[i].y, image[i].r, image[i].g, image[i].b);

		pEnv->SetObjectArrayElement(jImage, i, p);
	}

	return jImage;
}


int main() {
	return 0;
}